#include <chrono>
#include <gtest/gtest.h>
#include <optional>
#include <src/host_sbs_adaptive_submission.h>
#include <src/host_sbs_conversion_work.h>
#include <src/platform/windows/capture_timing.h>
#include <src/video_encode_pacing.h>

namespace {
  using namespace std::chrono_literals;
  using models::engine_build_status;
  using platf::dxgi::detail::capture_wait_policy_t;
  using platf::dxgi::detail::local_presenter_retry_state_t;

  struct idle_image_event_t {
    std::chrono::nanoseconds last_wait {};

    std::optional<int> pop(std::chrono::nanoseconds wait) {
      last_wait = wait;
      return std::nullopt;
    }
  };

  struct static_source_t {
    models::host_sbs_conversion_work_t work {.pipeline_enabled = true};
    engine_build_status model_status = engine_build_status::building;
    local_presenter_retry_state_t presenter;
    idle_image_event_t images;
    unsigned status_reads = 0u;

    static_source_t() {
      // The initial real source was displayed flat and is now retained by both adapters.
      presenter.observe_source();
      presenter.record_converted();
      presenter.record_presented();
    }

    bool needs_service() {
      return work.needs_service([&]() {
        ++status_reads;
        return model_status;
      });
    }

    void expect_idle() {
      const bool service = needs_service();
      EXPECT_FALSE(service);
      EXPECT_FALSE(presenter.should_process(true, false, service));
      const capture_wait_policy_t wait {presenter.should_process(true, false, service)};
      EXPECT_EQ(wait.source_timeout() + wait.idle_backoff(), 210ms);
      video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, service);
      EXPECT_EQ(images.last_wait, 83ms);
    }

    void expect_service() {
      const bool service = needs_service();
      EXPECT_TRUE(service);
      EXPECT_TRUE(presenter.should_process(true, false, service));
      EXPECT_TRUE(presenter.should_convert(true, false, service));
      const capture_wait_policy_t wait {presenter.should_process(true, false, service)};
      EXPECT_EQ(wait.source_timeout() + wait.idle_backoff(), 6ms);
      video::detail::wait_for_encode_image(images, 83ms, 16ms, true, false, service);
      EXPECT_EQ(images.last_wait, 16ms);
    }
  };

  TEST(HostSbsConversionWorkTest, GlobalPreparationReadyStartsStaticSourceThenReturnsToIdle) {
    static_source_t source;
    source.model_status = engine_build_status::unknown;
    source.expect_idle();
    source.model_status = engine_build_status::building;
    source.expect_idle();

    // No new capture or pipeline event accompanies global readiness. Both adapters notice it
    // when the existing idle wait ends and re-evaluate common conversion demand.
    source.model_status = engine_build_status::ready;
    source.expect_service();
    source.work.pipeline_build_pending = true;
    const auto reads_before_build = source.status_reads;
    EXPECT_FALSE(source.needs_service());
    EXPECT_EQ(source.status_reads, reads_before_build);

    // Per-device construction retains its separate readiness event; pending construction alone
    // does not make the converter redraw. Local capture may still shorten its construction wait.
    EXPECT_TRUE(source.presenter.should_convert(true, true, source.needs_service()));
    video::detail::wait_for_encode_image(source.images, 83ms, 16ms, true, true, source.needs_service());
    EXPECT_EQ(source.images.last_wait, 0ns);
    source.work.pipeline_build_pending = false;
    source.work.estimator_present = true;
    source.work.depth_completion_pending = true;
    source.expect_service();
    source.work.depth_completion_pending = false;
    source.presenter.record_converted();
    source.presenter.record_presented();
    source.expect_idle();
    EXPECT_EQ(source.status_reads, reads_before_build);
  }

  TEST(HostSbsConversionWorkTest, FailedPreparationIsRetiredOnceAndNewPipelineCanRetry) {
    static_source_t source;
    source.expect_idle();
    source.model_status = engine_build_status::failed;
    source.expect_service();

    // ensure_depth_estimator() publishes failure and latches the renderer flat. No separate
    // consumed-status flag can survive into a subsequent output/session initialization.
    source.work.pipeline_enabled = false;
    const auto reads_after_failure = source.status_reads;
    for (int timeout = 0; timeout < 3; ++timeout) {
      source.expect_idle();
    }
    EXPECT_EQ(source.status_reads, reads_after_failure);

    source.work.pipeline_enabled = true;
    source.expect_service();  // A new session observes even an unchanged failed model status.
    source.model_status = engine_build_status::building;
    source.expect_idle();
    source.model_status = engine_build_status::ready;
    source.expect_service();
  }

  TEST(HostSbsConversionWorkTest, CompletedOpaqueOwnershipDoesNotKeepEitherAdapterBusy) {
    static_source_t source;
    source.work.estimator_present = true;
    models::gpu_adaptive_transaction_policy_t ownership;
    const auto request = ownership.make_request(2u, true, false, 1u, 1000u);
    ASSERT_EQ(ownership.record_submission(2u, request, false, true), models::gpu_adaptive_submission_class_e::gpu_undecided);
    source.work.depth_completion_pending = true;
    source.expect_service();

    // Retire and present the exact transaction without exposing its GPU branch. The ownership
    // watermark stays live for the next changed source, while the static source becomes idle.
    source.work.depth_completion_pending = false;
    source.presenter.record_converted();
    source.presenter.record_presented();
    ASSERT_TRUE(ownership.active());
    for (int timeout = 0; timeout < 3; ++timeout) {
      source.expect_idle();
      EXPECT_EQ(ownership.conditional_frame_id(), 2u);
    }
    EXPECT_FALSE(ownership.make_request(3u, true, false, 1u, 2000u).authorize_gpu_undecided_reuse);
    EXPECT_TRUE(ownership.make_request(3u, true, true, 2u, 2000u).authorize_gpu_undecided_reuse);

    source.work.authority_reprocess_pending = true;
    source.expect_service();
    source.work.authority_reprocess_pending = false;
    source.work.dump_pending = true;
    source.expect_service();
    source.work.dump_pending = false;
    source.expect_idle();
    EXPECT_EQ(source.status_reads, 0u);
  }

  TEST(HostSbsConversionWorkTest, DisabledPipelineAndMissingSourceCannotStartStaticConversion) {
    static_source_t source;
    source.model_status = engine_build_status::ready;
    source.work.pipeline_enabled = false;
    source.expect_idle();
    EXPECT_EQ(source.status_reads, 0u);

    source.work.pipeline_enabled = true;
    const bool service = source.needs_service();
    ASSERT_TRUE(service);
    EXPECT_FALSE(source.presenter.should_process(false, false, service));
    EXPECT_FALSE(source.presenter.should_convert(false, false, service));
    video::detail::wait_for_encode_image(source.images, 83ms, 16ms, false, false, service);
    EXPECT_EQ(source.images.last_wait, 83ms);
  }
}  // namespace
