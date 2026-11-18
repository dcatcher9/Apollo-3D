/** Actual converter poll wiring and its existing owner-loop consumers, without D3D startup. */
#include <gtest/gtest.h>

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
  #include "src/config.h"
  #include "src/platform/windows/capture_timing.h"
  #include "src/platform/windows/conversion_poll_test.h"
  #include "src/thread_safe.h"
  #include "src/video_encode_pacing.h"

  #include <array>
  #include <chrono>
  #include <cstdlib>
  #include <future>
  #include <memory>
  #include <optional>
  #include <stdexcept>
  #include <string>
  #include <vector>

namespace {
  using namespace std::chrono_literals;
  using status_t = models::engine_build_status;
  using build_t = platf::dxgi::testing::conversion_poll_build_e;
  using inputs_t = platf::dxgi::testing::conversion_poll_inputs_t;
  using snapshot_t = platf::dxgi::testing::conversion_poll_snapshot_t;
  using probe_t = platf::dxgi::testing::conversion_poll_probe_t;
  using image_t = std::shared_ptr<int>;

  class diagnostic_file_scope_t {
  public:
    diagnostic_file_scope_t():
        saved_diagnostics_(config::sunshine.diagnostics_enabled) {
      if (const char *value = std::getenv("APOLLO_SBS_DUMP")) {
        saved_directory_ = value;
      }
      if (_putenv_s("APOLLO_SBS_DUMP", "") != 0) {
        throw std::runtime_error("Cannot isolate the CPU conversion-poll dump environment");
      }
      config::sunshine.diagnostics_enabled = false;
    }

    ~diagnostic_file_scope_t() {
      config::sunshine.diagnostics_enabled = saved_diagnostics_;
      EXPECT_EQ(_putenv_s("APOLLO_SBS_DUMP", saved_directory_ ? saved_directory_->c_str() : ""), 0);
    }

  private:
    bool saved_diagnostics_;
    std::optional<std::string> saved_directory_;
  };

  std::shared_future<std::vector<std::wstring>> ready_registry() {
    std::promise<std::vector<std::wstring>> census;
    auto future = census.get_future().share();
    census.set_value({});
    return future;
  }

  snapshot_t sample(probe_t &probe, const inputs_t &inputs, status_t status) {
    const auto result = probe.observe(inputs, [status] {
      return status;
    });
    EXPECT_FALSE(result.estimator_present);
    EXPECT_FALSE(result.device_present);
    EXPECT_FALSE(result.context_present);
    EXPECT_LE(result.model_status_reads, 1u);
    return result;
  }

  class ConversionPollActualMethodTest: public testing::Test {
  private:
    // This member is constructed before test-body probes and restored after their teardown.
    diagnostic_file_scope_t diagnostic_files_;
  };

  TEST_F(ConversionPollActualMethodTest, RegistryAbsentAndPendingUseIndependentModelStatus) {
    for (const bool present : {false, true}) {
      SCOPED_TRACE(present);
      std::promise<std::vector<std::wstring>> census;
      std::optional<std::shared_future<std::vector<std::wstring>>> future;
      if (present) {
        future = census.get_future().share();
      }
      probe_t probe {std::move(future)};
      for (const auto status : {status_t::unknown, status_t::building, status_t::ready, status_t::failed}) {
        SCOPED_TRACE(static_cast<int>(status));
        const auto result = sample(probe, {}, status);
        EXPECT_EQ(result.registry_present, present);
        EXPECT_FALSE(result.registry_ready);
        EXPECT_FALSE(result.registry_ready_observed);
        EXPECT_EQ(result.needs_conversion, status == status_t::ready || status == status_t::failed);
        EXPECT_EQ(result.model_status_reads, 1u);
      }
    }
  }

  TEST_F(ConversionPollActualMethodTest, ReadyRegistryPublishesObservedPollAndWaitCountsBeforeEstimator) {
    std::promise<std::vector<std::wstring>> census;
    probe_t probe {census.get_future().share()};
    ASSERT_FALSE(sample(probe, {}, status_t::building).registry_ready);
    census.set_value({});
    video::detail::latest_encode_source_t<image_t> source;
    const auto image = std::make_shared<int>(17);
    source.observe(image);
    source.converted();
    safe::event_t<image_t> images;
    platf::dxgi::detail::local_presenter_retry_state_t presenter;
    constexpr auto frame = 11111111ns;
    constexpr auto idle = 55555555ns;
    unsigned samples = 0, requests = 0, model_reads = 0, presenter_requests = 0;
    std::int64_t remote_wait_ns = 0, local_wait_ms = 0;
    for (const auto build : {build_t::absent, build_t::pending, build_t::ready}) {
      for (const auto status : {status_t::unknown, status_t::building}) {
        for (unsigned turn = 0; turn < 100; ++turn) {
          const auto result = sample(probe, {.build = build}, status);
          ASSERT_TRUE(result.registry_ready);
          EXPECT_FALSE(result.registry_ready_observed);
          EXPECT_EQ(result.build_valid, build != build_t::absent);
          EXPECT_EQ(result.build_ready, build == build_t::ready);
          ++samples;
          requests += result.needs_conversion;
          model_reads += result.model_status_reads;
          const auto wait = video::detail::encode_image_wait(idle, frame, true, false, result.needs_conversion);
          remote_wait_ns += wait.count();
          images.raise(source.latest());
          EXPECT_EQ(video::detail::pop_encode_image(images, wait, nullptr), image);
          EXPECT_FALSE(images.pop(0ns));
          EXPECT_EQ(source.latest(), image);
          EXPECT_FALSE(source.pending());
          const bool pending = presenter.should_process(true, false, result.needs_conversion);
          EXPECT_EQ(presenter.should_convert(true, false, result.needs_conversion), pending);
          presenter_requests += pending;
          const platf::dxgi::detail::capture_wait_policy_t policy {pending};
          local_wait_ms += (policy.source_timeout() + policy.idle_backoff()).count();
        }
      }
    }
    EXPECT_EQ(samples, 600u);
    // Common observational body: preserve the old result, then compare the real treatment.
    // These are actual method decisions/requested wait bounds, never elapsed throughput.
    RecordProperty("actual_poll_samples", static_cast<int>(samples));
    RecordProperty("conversion_requests", static_cast<int>(requests));
    RecordProperty("model_status_reads", static_cast<int>(model_reads));
    RecordProperty("presenter_requests", static_cast<int>(presenter_requests));
    RecordProperty("remote_requested_wait_total_ns", std::to_string(remote_wait_ns));
    RecordProperty("local_capture_policy_total_ms", std::to_string(local_wait_ms));
  }

  TEST_F(ConversionPollActualMethodTest, ObservedRegistryCannotSuppressGlobalReadyOrFailed) {
    probe_t probe {ready_registry()};
    for (const auto status : {status_t::unknown, status_t::building, status_t::ready, status_t::failed}) {
      const auto result = sample(probe, {.registry_ready_observed = true}, status);
      EXPECT_TRUE(result.registry_ready);
      EXPECT_TRUE(result.registry_ready_observed);
      EXPECT_EQ(result.needs_conversion, status == status_t::ready || status == status_t::failed);
      EXPECT_EQ(result.model_status_reads, 1u);
    }
  }

  TEST_F(ConversionPollActualMethodTest, PipelineFutureValidityKeepsItsIndependentCompletionWake) {
    std::promise<std::vector<std::wstring>> census;
    probe_t probe {census.get_future().share()};
    for (const auto build : {build_t::pending, build_t::ready}) {
      for (const auto status : {status_t::ready, status_t::failed}) {
        const auto result = sample(probe, {.build = build}, status);
        EXPECT_TRUE(result.build_valid);
        EXPECT_EQ(result.build_ready, build == build_t::ready);
        EXPECT_FALSE(result.needs_conversion);
        EXPECT_EQ(result.model_status_reads, 0u);
      }
    }
    // The real completed-build event is a separate owner-loop input. It cuts an actual empty
    // image-event pop to zero even while the converter still owns its ready-but-unconsumed future.
    safe::event_t<bool> pipeline_ready;
    pipeline_ready.raise(true);
    safe::event_t<image_t> images;
    const auto wait = video::detail::encode_image_wait(83ms, 16ms, true, pipeline_ready.peek(), false);
    EXPECT_EQ(wait, 0ns);
    EXPECT_FALSE(video::detail::pop_encode_image(images, wait, nullptr));
    platf::dxgi::detail::local_presenter_retry_state_t presenter;
    EXPECT_TRUE(presenter.should_process(true, pipeline_ready.peek(), false));
    EXPECT_TRUE(presenter.should_convert(true, pipeline_ready.peek(), false));
  }

  TEST_F(ConversionPollActualMethodTest, DepthAndAuthorityPendingCausesRemainIndependent) {
    std::promise<std::vector<std::wstring>> census;
    probe_t probe {census.get_future().share()};
    const std::array<inputs_t, 3> causes {{
      {.build = build_t::pending, .depth_completion_pending = true},
      {.build = build_t::pending, .depth_authority_reprocess_pending = true},
      {.build = build_t::pending, .subtitle_context_reprocess_pending = true},
    }};
    for (const auto &inputs : causes) {
      const auto result = sample(probe, inputs, status_t::building);
      EXPECT_TRUE(result.needs_conversion);
      EXPECT_EQ(result.model_status_reads, 0u);
    }
  }

  TEST_F(ConversionPollActualMethodTest, DumpButtonRequestsRetainedSourceWithoutModelRead) {
    probe_t probe {std::nullopt};
    const auto requested = sample(probe, {.dump_button_requested = true}, status_t::building);
    EXPECT_TRUE(requested.needs_conversion);
    EXPECT_EQ(requested.model_status_reads, 0u);
    const auto cleared = sample(probe, {}, status_t::building);
    EXPECT_FALSE(cleared.needs_conversion);
    EXPECT_EQ(cleared.model_status_reads, 1u);
  }

  TEST_F(ConversionPollActualMethodTest, GammaRequestAndDueWhiteQueryRemainIndependent) {
    probe_t probe {std::nullopt};
    const auto gamma = sample(probe, {.stream_gamma_conversion_pending = true}, status_t::building);
    EXPECT_TRUE(gamma.needs_conversion);
    EXPECT_EQ(gamma.model_status_reads, 0u);
    const auto white_due = sample(probe, {.white_query_due = true}, status_t::building);
    EXPECT_TRUE(white_due.needs_conversion);
    EXPECT_EQ(white_due.model_status_reads, 0u);
    const auto white_future = sample(probe, {.white_query_due = false}, status_t::building);
    EXPECT_FALSE(white_future.needs_conversion);
    EXPECT_EQ(white_future.model_status_reads, 1u);
  }

  TEST_F(ConversionPollActualMethodTest, DisabledDepthPipelineDoesNotPollOrReadModel) {
    probe_t probe {ready_registry()};
    const std::array<inputs_t, 2> disabled {{
      {.mode = video::SBS_OFF},
      {.renderer = models::host_sbs_renderer_e::failed_flat},
    }};
    for (const auto &inputs : disabled) {
      for (const auto status : {status_t::unknown, status_t::building, status_t::ready, status_t::failed}) {
        const auto result = sample(probe, inputs, status);
        EXPECT_FALSE(result.needs_conversion);
        EXPECT_EQ(result.model_status_reads, 0u);
        EXPECT_TRUE(result.registry_ready);
        EXPECT_FALSE(result.registry_ready_observed);
      }
    }
  }
}  // namespace
#endif

#if defined(_WIN32) && defined(SUNSHINE_TESTS)
namespace {
  TEST_F(ConversionPollActualMethodTest, ReadyRegistryWithoutEstimatorDoesNotConvertDuringUnknownOrBuilding) {
    probe_t probe {ready_registry()};
    for (const auto build : {build_t::absent, build_t::pending, build_t::ready}) {
      for (const auto status : {status_t::unknown, status_t::building}) {
        SCOPED_TRACE(static_cast<int>(build));
        SCOPED_TRACE(static_cast<int>(status));
        for (unsigned turn = 0; turn < 32; ++turn) {
          const auto result = sample(probe, {.build = build}, status);
          EXPECT_TRUE(result.registry_ready);
          EXPECT_FALSE(result.registry_ready_observed);
          EXPECT_FALSE(result.needs_conversion);
          EXPECT_EQ(result.build_valid, build != build_t::absent);
          EXPECT_EQ(result.build_ready, build == build_t::ready);
          EXPECT_EQ(result.model_status_reads, build == build_t::absent ? 1u : 0u);
        }
      }
    }
  }

  TEST_F(ConversionPollActualMethodTest, GlobalModelOutcomeAndBuildCompletionStillServiceReadyRegistrySource) {
    probe_t probe {ready_registry()};
    for (const auto status : {status_t::ready, status_t::failed}) {
      SCOPED_TRACE(static_cast<int>(status));
      const auto global = sample(probe, {}, status);
      EXPECT_TRUE(global.registry_ready);
      EXPECT_FALSE(global.registry_ready_observed);
      EXPECT_TRUE(global.needs_conversion);
      EXPECT_EQ(global.model_status_reads, 1u);
      EXPECT_EQ(video::detail::encode_image_wait(83ms, 16ms, true, false, global.needs_conversion), 16ms);

      for (const auto build : {build_t::pending, build_t::ready}) {
        const auto result = sample(probe, {.build = build}, status);
        EXPECT_TRUE(result.build_valid);
        EXPECT_EQ(result.build_ready, build == build_t::ready);
        EXPECT_FALSE(result.registry_ready_observed);
        EXPECT_FALSE(result.needs_conversion);
        EXPECT_EQ(result.model_status_reads, 0u);
        if (build == build_t::ready) {
          // An actual event object supplies the independent owner-loop completion input.
          // A ready, unconsumed future is never asserted to make the whole owner loop idle.
          safe::event_t<bool> completion;
          completion.raise(true);
          const auto wait = video::detail::encode_image_wait(83ms, 16ms, true, completion.peek(), result.needs_conversion);
          EXPECT_EQ(wait, 0ns);
          safe::event_t<image_t> images;
          EXPECT_FALSE(video::detail::pop_encode_image(images, wait, nullptr));
          platf::dxgi::detail::local_presenter_retry_state_t presenter;
          EXPECT_TRUE(presenter.should_process(true, completion.peek(), result.needs_conversion));
          EXPECT_TRUE(presenter.should_convert(true, completion.peek(), result.needs_conversion));
        }
      }
    }
  }
}  // namespace
#endif
