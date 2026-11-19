/** Opt-in fresh-process evidence for the actual live preparation owner and shared converter. */
#include <gtest/gtest.h>

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
  #include "src/config.h"
  #include "src/gpu_workload_arbiter.h"
  #include "src/live_model_preparation.h"
  #include "src/live_model_preparation_test.h"
  #include "src/model_manager.h"
  #include "src/platform/windows/model_preparation_converter_test.h"
  #include "src/video.h"
  #include "src/video_depth_estimator.h"

  #include <array>
  #include <chrono>
  #include <condition_variable>
  #include <cstdint>
  #include <cstdlib>
  #include <exception>
  #include <filesystem>
  #include <fstream>
  #include <mutex>
  #include <nlohmann/json.hpp>
  #include <optional>
  #include <stdexcept>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <utility>
  #include <vector>

namespace {
  using namespace std::chrono_literals;
  namespace probe = platf::dxgi::testing;
  namespace fs = std::filesystem;
  using json = nlohmann::json;
  using access_t = models::testing::live_model_preparation_test_access;
  using stage_t = models::testing::preparation_stage_e;
  using status_t = models::engine_build_status;
  using clock_t = std::chrono::steady_clock;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  bool enabled() {
    const auto *value = std::getenv("SUNSHINE_RUN_LIVE_PREPARATION_TESTS");
    return value && std::string_view(value) == "1";
  }

  struct plan_t {
    probe::model_preparation_adapter_t adapter;
    fs::path output;
    std::string preparation_adapter_name;
    json input;
  };

  plan_t read_plan() {
    const auto *value = std::getenv("SUNSHINE_LIVE_PREPARATION_PLAN");
    require(value && *value, "SUNSHINE_LIVE_PREPARATION_PLAN is required");
    const fs::path path(value);
    require(path.is_absolute() && fs::is_regular_file(path), "The preparation plan must be an absolute regular file");
    std::ifstream stream(path, std::ios::binary);
    const auto input = json::parse(stream);
    require(input.at("schema").get<unsigned>() == 1u, "Unsupported preparation plan schema");
    const auto &adapter = input.at("adapter");
    plan_t plan {
      .adapter = {
        .luid_low = adapter.at("luid_low").get<std::uint32_t>(),
        .luid_high = adapter.at("luid_high").get<std::int32_t>(),
        .vendor_id = adapter.at("vendor_id").get<std::uint32_t>(),
        .device_id = adapter.at("device_id").get<std::uint32_t>(),
      },
      .output = fs::path(input.at("output_root").get<std::string>()),
      .preparation_adapter_name = input.at("preparation_adapter_name").get<std::string>(),
      .input = input,
    };
    require(plan.output.is_absolute() && !fs::exists(plan.output), "The fixture requires a new absolute output directory");
    require(fs::is_directory(plan.output.parent_path()), "The output parent must already exist");
    require(plan.preparation_adapter_name.empty(), "This fixture requires the explicitly censused default CUDA device, not an ambiguous adapter name");
    require(fs::create_directory(plan.output), "Could not reserve the fixture output directory");
    return plan;
  }

  json identity(const probe::model_preparation_adapter_t &adapter) {
    return {{"luid_low", adapter.luid_low}, {"luid_high", adapter.luid_high}, {"vendor_id", adapter.vendor_id}, {"device_id", adapter.device_id}};
  }

  json memory(const probe::model_preparation_memory_t &sample) {
    const auto segment = [](const DXGI_QUERY_VIDEO_MEMORY_INFO &value) {
      return json {{"budget", value.Budget}, {"current_usage", value.CurrentUsage}, {"available_for_reservation", value.AvailableForReservation}, {"current_reservation", value.CurrentReservation}};
    };
    require(sample.local_status >= 0 && sample.nonlocal_status >= 0, "DXGI process-memory observation failed");
    return {{"adapter", identity(sample.adapter)}, {"description", sample.description}, {"node", 0}, {"local_status", sample.local_status}, {"nonlocal_status", sample.nonlocal_status}, {"local", segment(sample.local)}, {"nonlocal", segment(sample.nonlocal)}};
  }

  json observation(const probe::model_preparation_converter_snapshot_t &sample) {
    return {
      {"mode", sample.mode},
      {"model_status", static_cast<unsigned>(sample.model_status)},
      {"renderer", static_cast<unsigned>(sample.renderer)},
      {"conversions", sample.conversions},
      {"source_uploads", sample.source_uploads},
      {"source_stamp_ticks", sample.source_stamp_ticks},
      {"content_stamp_ticks", sample.content_stamp_ticks},
      {"rendered_stamp_ticks", sample.rendered_stamp_ticks},
      {"pipeline_ready", sample.pipeline_ready},
      {"build_pending", sample.build_pending},
      {"estimator_present", sample.estimator_present},
      {"estimator_valid", sample.estimator_valid},
      {"estimator_terminal", sample.estimator_terminal},
      {"v2_live_warp_seen", sample.v2_live_warp_seen},
      {"latest_lineage_authenticated", sample.latest_lineage_authenticated},
      {"latest_lineage_known_depth", sample.latest_lineage_known_depth},
      {"latest_lineage_receipt_present", sample.latest_lineage_receipt_present},
      {"latest_lineage_receipt_depth_cache_authorized", sample.latest_lineage_receipt_depth_cache_authorized},
      {"latest_lineage_receipt_matches_publication", sample.latest_lineage_receipt_matches_publication},
      {"latest_lineage_publication_frame_id", sample.latest_lineage_publication_frame_id},
      {"sbs_frame_sequence", sample.sbs_frame_sequence},
      {"flat_outputs", sample.flat_outputs},
      {"warped_outputs", sample.warped_outputs},
      {"repeat_outputs", sample.repeat_outputs},
      {"setup_watch_factory_calls", sample.setup_watch_factory_calls},
      {"setup_watch_tokens_alive", sample.setup_watch_tokens_alive},
      {"setup_watch_present", sample.setup_watch_present},
      {"receiver_present", sample.receiver_present},
      {"setup_watch_rect", {sample.setup_watch_rect.left, sample.setup_watch_rect.top, sample.setup_watch_rect.right, sample.setup_watch_rect.bottom}},
      {"last_draw_present", sample.last_draw_present},
      {"last_draw_flat", sample.last_draw_flat},
      {"last_draw_repeat", sample.last_draw_repeat},
      {"last_draw_publication_current", sample.last_draw_publication_current},
      {"last_draw_model_frame", sample.last_draw_model_frame},
      {"last_draw_observation_us", sample.last_draw_observation_us},
      {"last_draw_expected_work", sample.last_draw_expected_work},
      {"last_draw_host_subtitle_outcome", sample.last_draw_host_subtitle_outcome},
      {"last_draw_source_stamp_ticks", sample.last_draw_source_stamp_ticks},
      {"last_draw_content_stamp_ticks", sample.last_draw_content_stamp_ticks},
      {"last_draw_current_source_stamp_ticks", sample.last_draw_current_source_stamp_ticks},
      {"last_draw_current_content_stamp_ticks", sample.last_draw_current_content_stamp_ticks},
    };
  }

  json artifact(const probe::model_preparation_artifact_t &value) {
    require(fs::is_regular_file(value.path) && value.sha256.size() == 64u && models::file_sha256_hex(value.path) == value.sha256, "Converter artifact is absent or unauthenticated");
    return {{"path", value.path.string()}, {"sha256", value.sha256}, {"bytes", fs::file_size(value.path)}, {"width", value.width}, {"height", value.height}, {"row_bytes", value.row_bytes}};
  }

  void write_document(const fs::path &path, const json &report) {
    require(!fs::exists(path), "Refusing to overwrite a preparation report");
    std::ofstream stream(path, std::ios::binary);
    stream << report.dump(2) << '\n';
    stream.flush();
    require(stream.good(), "Could not write the preparation report");
  }

  void write_report(const plan_t &plan, const json &report) {
    write_document(plan.output / "preparation-report.json", report);
  }

  void checkpoint(const plan_t &plan, json &report, const char *phase) {
    report["active_phase"] = phase;
    write_document(plan.output / (std::string("checkpoint-") + phase + ".json"), report);
  }

  struct report_on_exit_t {
    const plan_t &plan;
    json &report;

    ~report_on_exit_t() noexcept {
      try {
        const bool exception = std::uncaught_exceptions() != 0;
        const bool failed = ::testing::Test::HasFailure();
        report["fixture_outcome"] = exception ? "exception" : failed ? "failed" :
                                                                       "completed";
        report["exit_context"] = {{"uncaught_exception", exception}, {"test_failed", failed}};
        write_report(plan, report);
      } catch (const std::exception &error) {
        try {
          ADD_FAILURE() << "Could not preserve preparation evidence: " << std::string_view(error.what()).substr(0u, 1024u);
        } catch (...) {
          // GoogleTest may be configured to throw after recording the failure. Never throw
          // another exception while unwinding the actual converter or preparation owner.
        }
      } catch (...) {
        try {
          ADD_FAILURE() << "Could not preserve preparation evidence: unknown write failure";
        } catch (...) {
        }
      }
    }
  };

  struct stages_t {
    std::mutex mutex;
    std::condition_variable changed;
    std::array<unsigned, 5> counts {};
    std::array<bool, 5> held {};
    std::array<bool, 5> released {};
    bool timed_out = false;

    static std::size_t index(stage_t stage) {
      return static_cast<std::size_t>(stage);
    }

    void observe(stage_t stage) noexcept {
      std::unique_lock lock(mutex);
      const auto slot = index(stage);
      ++counts[slot];
      changed.notify_all();
      if (held[slot] && !changed.wait_for(lock, 180s, [&] {
            return released[slot];
          })) {
        timed_out = true;
      }
    }

    bool wait(stage_t stage) {
      std::unique_lock lock(mutex);
      return changed.wait_for(lock, 180s, [&] {
        return counts[index(stage)] != 0u;
      });
    }

    void release(stage_t stage) {
      std::lock_guard lock(mutex);
      released[index(stage)] = true;
      changed.notify_all();
    }

    void release_all() {
      std::lock_guard lock(mutex);
      released.fill(true);
      changed.notify_all();
    }

    json snapshot() {
      std::lock_guard lock(mutex);
      return {{"worker_enter", counts[0]}, {"depth_building", counts[1]}, {"depth_finished", counts[2]}, {"ocr_finished", counts[3]}, {"worker_exit", counts[4]}, {"gate_timed_out", timed_out}};
    }
  };

  struct release_stages_t {
    stages_t &stages;

    ~release_stages_t() {
      stages.release_all();
    }
  };

  struct drain_t {
    bool drained = false;

    ~drain_t() {
      finish();
    }

    void finish() {
      if (!std::exchange(drained, true)) {
        video::drain_async_teardown_workers();
      }
    }
  };

  probe::model_preparation_converter_request_t source(const plan_t &plan) {
    constexpr std::uint32_t width = 1280u, height = 720u;
    probe::model_preparation_converter_request_t request;
    request.adapter = plan.adapter;
    request.width = width;
    request.height = height;
    request.output_root = plan.output;
    request.bgra.resize(width * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
      for (std::uint32_t x = 0; x < width; ++x) {
        const auto offset = 4u * (y * width + x);
        request.bgra[offset] = static_cast<std::uint8_t>((x * 3u + y) & 255u);
        request.bgra[offset + 1u] = static_cast<std::uint8_t>((x + 2u * y + ((x / 24u + y / 18u) & 1u) * 48u) & 255u);
        request.bgra[offset + 2u] = static_cast<std::uint8_t>((2u * x + y * 3u) & 255u);
        request.bgra[offset + 3u] = 255u;
      }
    }
    return request;
  }

  probe::model_preparation_converter_snapshot_t await_authenticated_static_output(probe::model_preparation_converter_t &converter) {
    const auto deadline = clock_t::now() + 180s;
    while (clock_t::now() < deadline) {
      auto sample = converter.observe();
      if (sample.estimator_terminal || sample.model_status == status_t::failed) {
        return sample;
      }
      if (sample.v2_live_warp_seen && sample.latest_lineage_authenticated && sample.last_draw_publication_current && sample.last_draw_present && !sample.last_draw_flat && !sample.last_draw_repeat) {
        return sample;
      }
      if (converter.needs_conversion_poll() || converter.pipeline_ready()) {
        (void) converter.convert_retained();
      } else {
        std::this_thread::sleep_for(1ms);
      }
    }
    return converter.observe();
  }

  void expect_flat_retained_source(const probe::model_preparation_converter_snapshot_t &sample) {
    EXPECT_TRUE(sample.last_draw_present);
    EXPECT_TRUE(sample.last_draw_flat);
    EXPECT_FALSE(sample.last_draw_publication_current);
    EXPECT_EQ(sample.last_draw_current_source_stamp_ticks, sample.source_stamp_ticks);
    EXPECT_EQ(sample.last_draw_current_content_stamp_ticks, sample.content_stamp_ticks);
  }

  json report_header(const plan_t &plan, std::string_view scenario) {
    return {{"schema", 1}, {"scenario", scenario}, {"plan", plan.input}, {"active_phase", "plan-accepted"}, {"first_output_oracle", "Actual current authenticated nonflat nonrepeat draw; late receipt-backed known_depth is recorded separately as future cache/reuse permission."}, {"preparation_adapter_policy", "default CUDA ordinal 0, bound by the parent CUDA-LUID census"}, {"fresh_process_required", true}, {"cache_claim", "existing cache state, not a cold-cache measurement"}, {"game_edge", "actual Game mono initialization/branch with provider transport unavailable and only setup-watch factory replaced by an owned inert token"}, {"phases", json::object()}};
  }

  void expect_completed_stages(const json &stages) {
    for (const auto *name : {"worker_enter", "depth_building", "depth_finished", "ocr_finished", "worker_exit"}) {
      EXPECT_EQ(stages.at(name).get<unsigned>(), 1u) << name;
    }
    EXPECT_FALSE(stages.at("gate_timed_out").get<bool>());
  }
}  // namespace

TEST(LiveModelPreparationGpuTest, ConstructorStateIdleFlatGameAndFirstAi) {
  if (!enabled()) {
    GTEST_SKIP() << "Set SUNSHINE_RUN_LIVE_PREPARATION_TESTS=1 and a fresh-process explicit adapter plan.";
  }
  const auto plan = read_plan();
  const auto model = video::host_sbs_v2_depth_model();
  auto report = report_header(plan, "idle-flat-game-first-ai");
  report_on_exit_t preserve_report {plan, report};
  RecordProperty("report", (plan.output / "preparation-report.json").string());
  report["initial_model_status"] = static_cast<unsigned>(models::tensorrt_model_prepare_status(model));
  report["phases"]["before_owner"] = memory(probe::query_model_preparation_memory(plan.adapter));
  checkpoint(plan, report, "before-owner");
  ASSERT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown) << "Run this case alone in a fresh process.";
  stages_t stages;
  std::optional<gpu_workload::lease_t> live_lease;
  drain_t drain;
  models::testing::preparation_hooks_t hooks;
  hooks.observe = [&](stage_t stage) {
    stages.observe(stage);
  };
  auto owner = access_t::create(model, plan.preparation_adapter_name, std::move(hooks));
  release_stages_t release {stages};
  // Observational join does not request work. It settles the original eager worker; a future
  // demand-owned implementation has no worker here and leaves the owner available for AI.
  access_t::join(*owner);
  report["phases"]["idle_after_constructor"] = {{"stages", stages.snapshot()}, {"model_status", static_cast<unsigned>(models::tensorrt_model_prepare_status(model))}, {"memory", memory(probe::query_model_preparation_memory(plan.adapter))}};
  checkpoint(plan, report, "idle-after-constructor");
  live_lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live_lease);
  {
    probe::model_preparation_converter_t converter(source(plan));
    const auto flat = converter.convert_retained();
    EXPECT_EQ(flat.mode, video::SBS_OFF);
    EXPECT_FALSE(flat.estimator_present);
    EXPECT_FALSE(flat.build_pending);
    report["source"] = artifact(converter.source_artifact());
    report["phases"]["flat"] = {{"observation", observation(flat)}, {"output", artifact(converter.snapshot_output("flat"))}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "flat");
    converter.set_mode(video::SBS_GAME_MONO);
    const auto game = converter.convert_retained();
    EXPECT_EQ(game.mode, video::SBS_GAME_MONO);
    EXPECT_TRUE(game.setup_watch_present);
    EXPECT_EQ(game.setup_watch_factory_calls, 1u);
    EXPECT_EQ(game.setup_watch_tokens_alive, 1u);
    EXPECT_FALSE(game.receiver_present);
    EXPECT_FALSE(game.estimator_present);
    EXPECT_FALSE(game.build_pending);
    report["phases"]["game_no_provider"] = {{"observation", observation(game)}, {"output", artifact(converter.snapshot_output("game-no-provider"))}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "game-no-provider");
    converter.set_mode(video::SBS_AI);
    (void) converter.convert_retained();
    report["active_phase"] = "awaiting-first-ai-output";
    const auto ready = await_authenticated_static_output(converter);
    report["phases"]["first_ai"] = {{"observation", observation(ready)}};
    checkpoint(plan, report, "first-ai-observation");
    report["phases"]["first_ai"]["output"] = artifact(converter.snapshot_output("first-ai"));
    report["phases"]["first_ai"]["memory"] = memory(converter.memory());
    checkpoint(plan, report, "first-ai-candidate");
    ASSERT_TRUE(ready.estimator_valid);
    ASSERT_FALSE(ready.estimator_terminal);
    ASSERT_TRUE(ready.v2_live_warp_seen);
    ASSERT_TRUE(ready.latest_lineage_authenticated);
    ASSERT_TRUE(ready.last_draw_publication_current);
    ASSERT_TRUE(ready.last_draw_present);
    ASSERT_FALSE(ready.last_draw_flat);
    ASSERT_FALSE(ready.last_draw_repeat);
    EXPECT_GT(ready.last_draw_model_frame, 0u);
    EXPECT_EQ(ready.last_draw_model_frame, ready.latest_lineage_publication_frame_id);
    EXPECT_EQ(ready.source_uploads, 1u);
    EXPECT_EQ(ready.source_stamp_ticks, flat.source_stamp_ticks);
    EXPECT_EQ(ready.content_stamp_ticks, flat.content_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_source_stamp_ticks, ready.source_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_content_stamp_ticks, ready.content_stamp_ticks);
    EXPECT_FALSE(ready.setup_watch_present);
    EXPECT_EQ(ready.setup_watch_tokens_alive, 0u);
    access_t::join(*owner);
    report["stages_after_ai"] = stages.snapshot();
    expect_completed_stages(report["stages_after_ai"]);
  }
  owner.reset();
  drain.finish();
  report["phases"]["after_owned_join_and_converter_drain"] = memory(probe::query_model_preparation_memory(plan.adapter));
  report["active_phase"] = "completed";
}

TEST(LiveModelPreparationGpuTest, StaticSourceObservesRealUnknownBuildingAndReadyBeforeOptionalOcr) {
  if (!enabled()) {
    GTEST_SKIP() << "Set SUNSHINE_RUN_LIVE_PREPARATION_TESTS=1 and a fresh-process explicit adapter plan.";
  }
  const auto plan = read_plan();
  const auto model = video::host_sbs_v2_depth_model();
  auto report = report_header(plan, "static-unknown-building-ready-before-ocr");
  report_on_exit_t preserve_report {plan, report};
  RecordProperty("report", (plan.output / "preparation-report.json").string());
  report["initial_model_status"] = static_cast<unsigned>(models::tensorrt_model_prepare_status(model));
  report["phases"]["before_owner"] = memory(probe::query_model_preparation_memory(plan.adapter));
  checkpoint(plan, report, "before-owner");
  ASSERT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown) << "Run this case alone in a fresh process.";
  stages_t stages;
  stages.held[stages_t::index(stage_t::worker_enter)] = true;
  stages.held[stages_t::index(stage_t::depth_building)] = true;
  stages.held[stages_t::index(stage_t::depth_finished)] = true;
  auto live_lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live_lease);
  drain_t drain;
  models::testing::preparation_hooks_t hooks;
  hooks.observe = [&](stage_t stage) {
    stages.observe(stage);
  };
  auto owner = access_t::create(model, plan.preparation_adapter_name, std::move(hooks));
  release_stages_t release {stages};
  {
    probe::model_preparation_converter_t converter(source(plan));
    converter.set_mode(video::SBS_AI);
    const auto unknown = converter.observe();
    EXPECT_EQ(unknown.model_status, status_t::unknown);
    EXPECT_FALSE(unknown.estimator_present);
    EXPECT_FALSE(unknown.build_pending);
    EXPECT_FALSE(converter.needs_conversion_poll());
    report["source"] = artifact(converter.source_artifact());
    report["phases"]["held_unknown"] = {{"observation", observation(unknown)}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "held-unknown");
    const auto accepted = converter.convert_retained();
    report["phases"]["accepted_first_ai_request"] = observation(accepted);
    checkpoint(plan, report, "accepted-first-ai-request");
    ASSERT_TRUE(stages.wait(stage_t::worker_enter));
    // Eager preparation has not entered the real setter yet. A demand-owned service may
    // publish building synchronously when it accepts this first conversion's request.
    EXPECT_TRUE(accepted.model_status == status_t::unknown || accepted.model_status == status_t::building);
    EXPECT_FALSE(accepted.estimator_present);
    EXPECT_FALSE(accepted.build_pending);
    expect_flat_retained_source(accepted);
    stages.release(stage_t::worker_enter);
    report["active_phase"] = "waiting-depth-building";
    ASSERT_TRUE(stages.wait(stage_t::depth_building));
    EXPECT_EQ(models::tensorrt_model_prepare_status(model), status_t::building);
    const auto building = converter.convert_retained();
    EXPECT_EQ(building.model_status, status_t::building);
    EXPECT_FALSE(building.estimator_present);
    EXPECT_FALSE(building.build_pending);
    expect_flat_retained_source(building);
    EXPECT_FALSE(converter.needs_conversion_poll());
    report["phases"]["held_building"] = {{"observation", observation(building)}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "held-building");
    stages.release(stage_t::depth_building);
    report["active_phase"] = "waiting-depth-finished";
    ASSERT_TRUE(stages.wait(stage_t::depth_finished));
    report["phases"]["depth_finished"] = {{"observation", observation(converter.observe())}, {"stages", stages.snapshot()}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "depth-finished");
    ASSERT_EQ(models::tensorrt_model_prepare_status(model), status_t::ready);
    EXPECT_EQ(stages.snapshot().at("ocr_finished").get<unsigned>(), 0u);
    EXPECT_TRUE(converter.needs_conversion_poll());
    const auto construction = converter.convert_retained();
    EXPECT_TRUE(construction.build_pending || construction.estimator_present);
    EXPECT_EQ(construction.source_stamp_ticks, unknown.source_stamp_ticks);
    EXPECT_EQ(construction.content_stamp_ticks, unknown.content_stamp_ticks);
    report["phases"]["depth_ready_before_ocr"] = {{"observation", observation(construction)}, {"stages", stages.snapshot()}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "depth-ready-before-ocr");
    stages.release(stage_t::depth_finished);
    report["active_phase"] = "awaiting-static-ai-output";
    const auto ready = await_authenticated_static_output(converter);
    report["phases"]["static_ai_ready"] = {{"observation", observation(ready)}};
    checkpoint(plan, report, "static-ai-observation");
    report["phases"]["static_ai_ready"]["output"] = artifact(converter.snapshot_output("static-ai-ready"));
    report["phases"]["static_ai_ready"]["memory"] = memory(converter.memory());
    checkpoint(plan, report, "static-ai-candidate");
    ASSERT_TRUE(ready.estimator_valid);
    ASSERT_FALSE(ready.estimator_terminal);
    ASSERT_TRUE(ready.v2_live_warp_seen);
    ASSERT_TRUE(ready.latest_lineage_authenticated);
    ASSERT_TRUE(ready.last_draw_publication_current);
    ASSERT_TRUE(ready.last_draw_present);
    ASSERT_FALSE(ready.last_draw_flat);
    ASSERT_FALSE(ready.last_draw_repeat);
    EXPECT_GT(ready.last_draw_model_frame, 0u);
    EXPECT_EQ(ready.last_draw_model_frame, ready.latest_lineage_publication_frame_id);
    EXPECT_EQ(ready.source_uploads, 1u);
    EXPECT_EQ(ready.source_stamp_ticks, unknown.source_stamp_ticks);
    EXPECT_EQ(ready.content_stamp_ticks, unknown.content_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_source_stamp_ticks, ready.source_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_content_stamp_ticks, ready.content_stamp_ticks);
    access_t::join(*owner);
    report["stages_after_join"] = stages.snapshot();
    expect_completed_stages(report["stages_after_join"]);
  }
  owner.reset();
  drain.finish();
  report["active_phase"] = "completed";
}

TEST(LiveModelPreparationGpuTest, RejectedCanonicalIdentityPublishesRealFailureForTheStaticSource) {
  if (!enabled()) {
    GTEST_SKIP() << "Set SUNSHINE_RUN_LIVE_PREPARATION_TESTS=1 and a fresh-process explicit adapter plan.";
  }
  const auto plan = read_plan();
  const auto canonical = video::host_sbs_v2_depth_model();
  auto invalid = canonical;
  invalid.url += "#deliberately-invalid-preparation-fixture";
  auto report = report_header(plan, "real-canonical-identity-failure");
  report_on_exit_t preserve_report {plan, report};
  RecordProperty("report", (plan.output / "preparation-report.json").string());
  report["failure_scope"] = "Wrong canonical URL is rejected by the real resolver before depth filesystem/CUDA work; no cache or asset is modified to create this failure.";
  report["initial_model_status"] = static_cast<unsigned>(models::tensorrt_model_prepare_status(canonical));
  checkpoint(plan, report, "before-owner");
  ASSERT_EQ(models::tensorrt_model_prepare_status(canonical), status_t::unknown) << "Run this case alone in a fresh process.";
  stages_t stages;
  stages.held[stages_t::index(stage_t::worker_enter)] = true;
  stages.held[stages_t::index(stage_t::depth_building)] = true;
  stages.held[stages_t::index(stage_t::depth_finished)] = true;
  auto live_lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live_lease);
  drain_t drain;
  models::testing::preparation_hooks_t hooks;
  hooks.observe = [&](stage_t stage) {
    stages.observe(stage);
  };
  auto owner = access_t::create(invalid, plan.preparation_adapter_name, std::move(hooks));
  release_stages_t release {stages};
  {
    probe::model_preparation_converter_t converter(source(plan));
    converter.set_mode(video::SBS_AI);
    const auto unknown = converter.observe();
    EXPECT_EQ(unknown.model_status, status_t::unknown);
    EXPECT_FALSE(converter.needs_conversion_poll());
    const auto accepted = converter.convert_retained();
    report["source"] = artifact(converter.source_artifact());
    report["phases"]["accepted_first_ai_request"] = observation(accepted);
    checkpoint(plan, report, "accepted-first-ai-request");
    ASSERT_TRUE(stages.wait(stage_t::worker_enter));
    EXPECT_TRUE(accepted.model_status == status_t::unknown || accepted.model_status == status_t::building);
    EXPECT_FALSE(accepted.estimator_present);
    EXPECT_FALSE(accepted.build_pending);
    expect_flat_retained_source(accepted);
    stages.release(stage_t::worker_enter);
    report["active_phase"] = "waiting-depth-building";
    ASSERT_TRUE(stages.wait(stage_t::depth_building));
    EXPECT_EQ(models::tensorrt_model_prepare_status(canonical), status_t::building);
    stages.release(stage_t::depth_building);
    report["active_phase"] = "waiting-depth-finished";
    ASSERT_TRUE(stages.wait(stage_t::depth_finished));
    report["phases"]["depth_finished"] = {{"observation", observation(converter.observe())}, {"stages", stages.snapshot()}, {"memory", memory(converter.memory())}};
    checkpoint(plan, report, "depth-finished");
    ASSERT_EQ(models::tensorrt_model_prepare_status(canonical), status_t::failed);
    EXPECT_TRUE(converter.needs_conversion_poll());
    const auto failed = converter.convert_retained();
    report["phases"]["terminal_flat"] = {{"observation", observation(failed)}, {"stages", stages.snapshot()}};
    checkpoint(plan, report, "terminal-flat-observation");
    report["phases"]["terminal_flat"]["output"] = artifact(converter.snapshot_output("terminal-flat"));
    report["phases"]["terminal_flat"]["memory"] = memory(converter.memory());
    checkpoint(plan, report, "terminal-flat");
    EXPECT_EQ(failed.model_status, status_t::failed);
    EXPECT_FALSE(failed.estimator_present);
    EXPECT_FALSE(failed.build_pending);
    EXPECT_FALSE(failed.v2_live_warp_seen);
    EXPECT_EQ(failed.renderer, models::host_sbs_renderer_e::failed_flat);
    expect_flat_retained_source(failed);
    EXPECT_FALSE(converter.needs_conversion_poll());
    EXPECT_EQ(failed.source_uploads, 1u);
    EXPECT_EQ(failed.source_stamp_ticks, unknown.source_stamp_ticks);
    EXPECT_EQ(failed.content_stamp_ticks, unknown.content_stamp_ticks);
    stages.release(stage_t::depth_finished);
    report["active_phase"] = "waiting-worker-exit";
    ASSERT_TRUE(stages.wait(stage_t::worker_exit));
    access_t::join(*owner);
    report["stages_after_join"] = stages.snapshot();
    expect_completed_stages(report["stages_after_join"]);
  }
  owner.reset();
  drain.finish();
  report["active_phase"] = "completed";
}

TEST(LiveModelPreparationGpuTest, ChangedHardDisabledSourceReusesDepthAndDrawsCurrentColour) {
  if (!enabled()) {
    GTEST_SKIP() << "Set SUNSHINE_RUN_LIVE_PREPARATION_TESTS=1 and an explicit adapter plan.";
  }
  const auto plan = read_plan();
  auto live_lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live_lease);
  drain_t drain;
  auto owner = access_t::create(video::host_sbs_v2_depth_model(), plan.preparation_adapter_name, {});
  auto request = source(plan);
  request.hard_disabled_dirty_source = true;
  auto changed = request.bgra;
  const RECT damage {320, 240, 352, 272};
  for (LONG y = damage.top; y != damage.bottom; ++y) {
    for (LONG x = damage.left; x != damage.right; ++x) {
      const auto offset = (std::size_t(y) * request.width + x) * 4u;
      for (unsigned channel = 0; channel != 3u; ++channel) {
        changed[offset + channel] ^= 4u;
      }
    }
  }
  {
    probe::model_preparation_converter_t converter(std::move(request));
    converter.set_mode(video::SBS_AI);
    const auto await_receipt = [&](std::chrono::seconds budget) {
      const auto deadline = std::chrono::steady_clock::now() + budget;
      while (std::chrono::steady_clock::now() < deadline) {
        const auto sample = converter.observe();
        if (sample.estimator_terminal || sample.model_status == status_t::failed ||
            (sample.receipt && sample.receipt->depth_cache_authorized() &&
             sample.latest_lineage_receipt_matches_publication && sample.last_draw_present &&
             !sample.last_draw_flat && sample.last_draw_publication_current &&
             sample.last_draw_model_frame == sample.receipt->expected.frame_id &&
             sample.last_draw_source_stamp_ticks == sample.source_stamp_ticks)) {
          return sample;
        }
        // A completed draw can precede the nonblocking receipt. Keep delivering this immutable
        // source while waiting for the converter to consume that receipt.
        (void) converter.convert_retained();
        std::this_thread::sleep_for(1ms);
      }
      return converter.observe();
    };
    const auto a = await_receipt(180s);
    ASSERT_TRUE(a.receipt && a.receipt->depth_cache_authorized()) << observation(a).dump();
    ASSERT_TRUE(a.receipt->expected.depth_only_history);
    ASSERT_TRUE(converter.drawn_source_matches_upload());
    const auto packed_a = converter.snapshot_output("dirty-a");
    converter.upload_dirty_source(changed, damage);
    const auto b = await_receipt(30s);
    ASSERT_TRUE(b.receipt && b.receipt->depth_cache_authorized()) << observation(b).dump();
    EXPECT_TRUE(b.latest_lineage_receipt_matches_publication);
    EXPECT_TRUE(b.last_admission_gpu_undecided);
    EXPECT_EQ(b.last_admission_optional_work, static_cast<std::uint32_t>(models::depth_optional_work_mode_e::disable_subtitle));
    EXPECT_GT(b.receipt->expected.frame_id, a.receipt->expected.frame_id);
    EXPECT_EQ(b.receipt->expected.expected_work, 0u);
    EXPECT_EQ(b.receipt->depth, models::host_sbs_gpu_completion_receipt::depth_disposition_e::reuse);
    EXPECT_EQ(b.receipt->depth_owner_frame_id, a.receipt->depth_owner_frame_id);
    EXPECT_EQ(b.receipt->depth_owner_timestamp_us, a.receipt->depth_owner_timestamp_us);
    EXPECT_EQ(b.receipt->subtitle, models::host_sbs_gpu_completion_receipt::subtitle_disposition_e::suppressed);
    EXPECT_TRUE(b.receipt->expected.depth_only_history);
    EXPECT_EQ(b.source_uploads, 2u);
    EXPECT_NE(b.content_stamp_ticks, a.content_stamp_ticks);
    EXPECT_EQ(b.last_draw_source_stamp_ticks, b.source_stamp_ticks);
    EXPECT_EQ(b.last_draw_current_content_stamp_ticks, b.content_stamp_ticks);
    EXPECT_TRUE(converter.drawn_source_matches_upload());
    EXPECT_NE(converter.snapshot_output("dirty-b").sha256, packed_a.sha256);
  }
  access_t::join(*owner);
  owner.reset();
  drain.finish();
}

#endif
