/** Matched opt-in real-GPU work-demand evidence through the production registry observer. */
#include <gtest/gtest.h>

#include "src/config.h"
#include "src/gpu_workload_arbiter.h"
#include "src/live_model_preparation.h"
#include "src/live_model_preparation_test.h"
#include "src/model_manager.h"
#include "src/platform/windows/model_preparation_converter_test.h"
#include "src/video.h"
#include "src/video_depth_estimator.h"
#include "src/video_encode_pacing.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
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
  using fixture_clock_t = std::chrono::steady_clock;

  void require(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  bool enabled() {
    const auto *value = std::getenv("SUNSHINE_RUN_REGISTRY_READINESS_GPU_TESTS");
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
      {"production_observer_path", sample.production_observer_path},
      {"subtitle_warp_ready", sample.subtitle_warp_ready},
      {"text_game_warp_ready", sample.text_game_warp_ready},
      {"registry_present", sample.registry_present},
      {"registry_ready", sample.registry_ready},
      {"registry_ready_observed", sample.registry_ready_observed},
      {"registry_factory_calls", sample.registry_factory_calls},
      {"registry_consumption_eligible", sample.registry_consumption_eligible},
      {"pipeline_enabled", sample.pipeline_enabled},
      {"depth_completion_pending", sample.depth_completion_pending},
      {"authority_reprocess_pending", sample.authority_reprocess_pending},
      {"subtitle_reprocess_pending", sample.subtitle_reprocess_pending},
      {"dump_pending", sample.dump_pending},
      {"gamma_pending", sample.gamma_pending},
      {"gamma_white_query_due", sample.gamma_white_query_due},
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
      {"last_draw_ema_motion_mask_present", sample.last_draw_ema_motion_mask_present},
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
    const auto deadline = fixture_clock_t::now() + 180s;
    while (fixture_clock_t::now() < deadline) {
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

  void expect_completed_stages(const json &stages) {
    for (const auto *name : {"worker_enter", "depth_building", "depth_finished", "ocr_finished", "worker_exit"}) {
      EXPECT_EQ(stages.at(name).get<unsigned>(), 1u) << name;
    }
    EXPECT_FALSE(stages.at("gate_timed_out").get<bool>());
  }
}  // namespace

TEST(HostSbsRegistryReadinessGpuTest, BuildingCensusWakeWaitsForTheActualObservationConsumer) {
  if (!enabled()) {
    GTEST_SKIP() << "Set SUNSHINE_RUN_REGISTRY_READINESS_GPU_TESTS=1 and supply the approved fresh-process preparation plan.";
  }
  const auto plan = read_plan();
  const auto model = video::host_sbs_v2_depth_model();
  json report {
    {"schema", 1},
    {"scenario", "production-observer-registry-work-demand"},
    {"plan", plan.input},
    {"active_phase", "plan-accepted"},
    {"fresh_process_required", true},
    {"work_oracle", "Fixed poll opportunities and authored keepalives; actual conversion/pacing observations, not measured FPS or walltime performance."},
    {"authority_oracle", "Real foreground observation with no output mapping: ordinary full-source geometry, no granted native-video/ROI authority."},
    {"content_oracle", "Exact raw source/flat/final BGRA8 hashes; current authenticated nonflat nonrepeat final draw. Receipt cache admission is observational."},
    {"encoder_scope", "RGB converter only: no input-copy, encoded-packet, network or bitrate claim."},
    {"phases", json::object()},
  };
  report_on_exit_t preserve_report {plan, report};
  RecordProperty("report", (plan.output / "preparation-report.json").string());
  report["initial_model_status"] = static_cast<unsigned>(models::tensorrt_model_prepare_status(model));
  report["initial_memory"] = memory(probe::query_model_preparation_memory(plan.adapter));
  checkpoint(plan, report, "before-owner");
  ASSERT_EQ(models::tensorrt_model_prepare_status(model), status_t::unknown) << "Run this case alone in a fresh process.";
  stages_t stages;
  stages.held[stages_t::index(stage_t::depth_building)] = true;
  stages.held[stages_t::index(stage_t::depth_finished)] = true;
  std::promise<std::vector<std::wstring>> census;
  auto census_result = census.get_future().share();
  auto live_lease = gpu_workload::try_acquire(gpu_workload::kind_e::live_stream);
  ASSERT_TRUE(live_lease);
  drain_t drain;
  models::testing::preparation_hooks_t hooks;
  hooks.observe = [&](stage_t stage) {
    stages.observe(stage);
  };
  auto owner = access_t::create(model, plan.preparation_adapter_name, std::move(hooks));
  {
    auto request = source(plan);
    request.production_observer_path = true;
    request.registry_executables = census_result;
    probe::model_preparation_converter_t converter(std::move(request));
    release_stages_t release {stages};
    converter.set_mode(video::SBS_AI);
    EXPECT_EQ(converter.observe().model_status, status_t::unknown);
    EXPECT_FALSE(converter.needs_conversion_poll());
    report["source"] = artifact(converter.source_artifact());
    const auto requested = converter.convert_retained();
    report["phases"]["first_request"] = observation(requested);
    checkpoint(plan, report, "first-request");
    ASSERT_TRUE(stages.wait(stage_t::depth_building));
    ASSERT_EQ(models::tensorrt_model_prepare_status(model), status_t::building);
    const auto pending = converter.observe();
    ASSERT_TRUE(pending.production_observer_path);
    ASSERT_TRUE(pending.registry_present);
    EXPECT_EQ(pending.registry_factory_calls, 1u);
    EXPECT_FALSE(pending.registry_ready);
    EXPECT_FALSE(pending.registry_ready_observed);
    EXPECT_FALSE(pending.registry_consumption_eligible);
    EXPECT_FALSE(pending.estimator_present);
    EXPECT_FALSE(pending.build_pending);
    EXPECT_TRUE(pending.pipeline_enabled);
    EXPECT_FALSE(pending.depth_completion_pending);
    EXPECT_FALSE(pending.authority_reprocess_pending);
    EXPECT_FALSE(pending.subtitle_reprocess_pending);
    EXPECT_FALSE(pending.dump_pending);
    EXPECT_FALSE(pending.gamma_pending);
    EXPECT_FALSE(pending.gamma_white_query_due);
    EXPECT_FALSE(converter.needs_conversion_poll());
    expect_flat_retained_source(pending);
    report["phases"]["building_pending_census"] = {{"observation", observation(pending)}, {"stages", stages.snapshot()}};
    checkpoint(plan, report, "building-pending-census");
    const auto initial_flat = converter.snapshot_output("building-initial-flat");
    report["phases"]["building_pending_census"]["output"] = artifact(initial_flat);

    // This is an actual ready empty census, not a fabricated ready or observed flag.
    census.set_value({});
    const auto ready_census = converter.observe();
    ASSERT_TRUE(ready_census.registry_ready);
    EXPECT_FALSE(ready_census.registry_ready_observed);
    EXPECT_EQ(ready_census.model_status, status_t::building);
    report["phases"]["building_ready_census"] = observation(ready_census);
    checkpoint(plan, report, "building-ready-census");

    const auto frame_interval = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / 90.0));
    const auto minimum_millifps = video::detail::minimum_fps_target(0.0, 90'000);
    const auto idle_interval = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1000.0 / minimum_millifps));
    unsigned poll_requests = 0, conversions_issued = 0, keepalives = 0;
    report["opportunities"] = json::array();
    for (unsigned index = 0; index != 32u; ++index) {
      const bool poll = converter.needs_conversion_poll();
      const bool keepalive = index % 5u == 4u;
      const auto wait = video::detail::encode_image_wait(idle_interval, frame_interval, true, converter.pipeline_ready(), poll);
      poll_requests += poll;
      keepalives += keepalive;
      if (poll || keepalive) {
        (void) converter.convert_retained();
        ++conversions_issued;
      }
      const auto actual = converter.observe();
      EXPECT_EQ(actual.model_status, status_t::building);
      EXPECT_FALSE(actual.estimator_present);
      EXPECT_FALSE(actual.build_pending);
      EXPECT_TRUE(actual.registry_ready);
      EXPECT_FALSE(actual.registry_ready_observed);
      EXPECT_FALSE(actual.registry_consumption_eligible);
      EXPECT_FALSE(actual.depth_completion_pending);
      EXPECT_FALSE(actual.authority_reprocess_pending);
      EXPECT_FALSE(actual.subtitle_reprocess_pending);
      EXPECT_FALSE(actual.dump_pending);
      EXPECT_FALSE(actual.gamma_pending);
      EXPECT_FALSE(actual.gamma_white_query_due);
      EXPECT_EQ(actual.source_uploads, 1u);
      EXPECT_EQ(actual.source_stamp_ticks, pending.source_stamp_ticks);
      EXPECT_EQ(actual.content_stamp_ticks, pending.content_stamp_ticks);
      expect_flat_retained_source(actual);
      report["opportunities"].push_back({{"index", index}, {"poll", poll}, {"authored_keepalive", keepalive}, {"actual_wait_ns", wait.count()}, {"observation", observation(actual)}});
    }
    const auto after_opportunities = converter.observe();
    EXPECT_EQ(keepalives, 6u);
    EXPECT_EQ(after_opportunities.conversions, pending.conversions + conversions_issued);
    report["phases"]["bounded_building_work"] = {{"poll_opportunities", 32}, {"poll_requests", poll_requests}, {"authored_keepalives", keepalives}, {"actual_conversions_issued", conversions_issued}, {"frame_interval_ns", frame_interval.count()}, {"idle_interval_ns", idle_interval.count()}, {"observation", observation(after_opportunities)}};
    checkpoint(plan, report, "bounded-building-work");
    const auto final_flat = converter.snapshot_output("building-final-flat");
    report["phases"]["bounded_building_work"]["output"] = artifact(final_flat);
    EXPECT_EQ(final_flat.sha256, initial_flat.sha256);

    stages.release(stage_t::depth_building);
    report["active_phase"] = "waiting-depth-finished";
    ASSERT_TRUE(stages.wait(stage_t::depth_finished));
    const auto model_ready = converter.observe();
    ASSERT_EQ(model_ready.model_status, status_t::ready);
    EXPECT_FALSE(model_ready.estimator_present);
    EXPECT_FALSE(model_ready.build_pending);
    EXPECT_FALSE(model_ready.registry_ready_observed);
    EXPECT_EQ(stages.snapshot().at("ocr_finished").get<unsigned>(), 0u);
    // Global readiness remains an independent reason to construct the device pipeline.
    EXPECT_TRUE(converter.needs_conversion_poll());
    const auto construction = converter.convert_retained();
    EXPECT_TRUE(construction.build_pending || construction.estimator_present);
    report["phases"]["depth_ready_before_ocr"] = {{"observation", observation(construction)}, {"stages", stages.snapshot()}};
    checkpoint(plan, report, "depth-ready-before-ocr");
    stages.release(stage_t::depth_finished);

    // Observe the actual device completion event before adoption. Registry readiness in the
    // control must not turn this fixture's event wait into an unpaced flat-draw loop.
    const bool poll_during_device_build = converter.needs_conversion_poll();
    const auto pipeline_deadline = fixture_clock_t::now() + 360s;
    while (!converter.pipeline_ready() && fixture_clock_t::now() < pipeline_deadline) {
      std::this_thread::sleep_for(1ms);
    }
    const bool pipeline_notified = converter.pipeline_ready();
    report["phases"]["device_event"] = {{"notified", pipeline_notified}, {"poll_during_device_build", poll_during_device_build}, {"observation", observation(converter.observe())}};
    checkpoint(plan, report, "device-event");
    ASSERT_TRUE(pipeline_notified);
    const auto adopted = converter.convert_retained();
    report["phases"]["actual_observer_adoption"] = observation(adopted);
    checkpoint(plan, report, "actual-observer-adoption");
    ASSERT_TRUE(adopted.estimator_present);
    ASSERT_TRUE(adopted.estimator_valid);
    EXPECT_TRUE(adopted.subtitle_warp_ready);
    EXPECT_FALSE(adopted.text_game_warp_ready);
    EXPECT_TRUE(adopted.registry_ready_observed);

    report["active_phase"] = "waiting-actual-authenticated-output";
    const auto ready = await_authenticated_static_output(converter);
    report["phases"]["actual_current_output"] = observation(ready);
    checkpoint(plan, report, "actual-current-observation");
    ASSERT_TRUE(ready.estimator_valid);
    ASSERT_FALSE(ready.estimator_terminal);
    ASSERT_TRUE(ready.v2_live_warp_seen);
    ASSERT_TRUE(ready.latest_lineage_authenticated);
    ASSERT_TRUE(ready.last_draw_present);
    ASSERT_FALSE(ready.last_draw_flat);
    ASSERT_FALSE(ready.last_draw_repeat);
    ASSERT_TRUE(ready.last_draw_publication_current);
    EXPECT_FALSE(ready.last_draw_ema_motion_mask_present);
    EXPECT_GT(ready.last_draw_model_frame, 0u);
    EXPECT_EQ(ready.last_draw_model_frame, ready.latest_lineage_publication_frame_id);
    EXPECT_TRUE(ready.production_observer_path);
    EXPECT_TRUE(ready.registry_ready);
    EXPECT_TRUE(ready.registry_consumption_eligible);
    EXPECT_TRUE(ready.registry_ready_observed) << "Only actual observe_subtitle_context may consume the census.";
    EXPECT_EQ(ready.registry_factory_calls, 1u);
    EXPECT_EQ(ready.source_uploads, 1u);
    EXPECT_EQ(ready.source_stamp_ticks, pending.source_stamp_ticks);
    EXPECT_EQ(ready.content_stamp_ticks, pending.content_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_source_stamp_ticks, ready.source_stamp_ticks);
    EXPECT_EQ(ready.last_draw_current_content_stamp_ticks, ready.content_stamp_ticks);
    report["phases"]["actual_current_output"]["output"] = artifact(converter.snapshot_output("actual-current-output"));
    report["phases"]["actual_current_output"]["memory"] = memory(converter.memory());
    checkpoint(plan, report, "actual-current-artifact");
    stages.release_all();
  }
  access_t::join(*owner);
  report["completed_stages"] = stages.snapshot();
  expect_completed_stages(report["completed_stages"]);
  owner.reset();
  drain.finish();
  report["final_memory"] = memory(probe::query_model_preparation_memory(plan.adapter));
  checkpoint(plan, report, "after-owned-drain");
}
