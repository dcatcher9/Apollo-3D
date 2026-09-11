#include <chrono>
#include <gtest/gtest.h>
#include <optional>
#include <src/host_sbs_conversion_work.h>
#include <src/platform/windows/capture_timing.h>
#include <src/video_encode_pacing.h>
#include <tests/fixtures/host_sbs_gpu_completion_receipt.h>

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

  TEST(HostSbsConversionWorkTest, CompletedOutputDoesNotKeepEitherAdapterBusyForItsReceipt) {
    static_source_t source;
    source.work.estimator_present = true;
    source.work.depth_completion_pending = true;
    source.expect_service();

    // Completed output needs no extra analysis. Its pending receipt is drained by ordinary
    // idle checks without requiring another conversion/root.
    source.work.depth_completion_pending = false;
    source.presenter.record_converted();
    source.presenter.record_presented();
    for (int timeout = 0; timeout < 3; ++timeout) {
      source.expect_idle();
    }

    source.work.authority_reprocess_pending = true;
    source.expect_service();
    source.work.authority_reprocess_pending = false;
    source.work.dump_pending = true;
    source.expect_service();
    source.work.dump_pending = false;
    source.expect_idle();
    EXPECT_EQ(source.status_reads, 0u);
  }

  TEST(HostSbsConversionWorkTest, StableNearReuseDoesNotCreateAnalysisWorkInEitherAdapter) {
    namespace fixture = host_sbs_gpu_completion_receipt_fixture;
    namespace receipt = models::host_sbs_gpu_completion_receipt;
    static_source_t source;
    source.work.estimator_present = true;
    const auto publication = fixture::expected();
    source.work.depth_completion_pending = true;
    source.expect_service();

    // B completed using A's joint analysis, then B stopped changing. Both adapters retain
    // their completed output indefinitely; only unfinished work is conversion demand.
    source.work.depth_completion_pending = false;
    source.presenter.record_converted();
    source.presenter.record_presented();
    source.expect_idle();  // A not-yet-drained receipt creates no new root either.
    const auto held = receipt::decode(fixture::snapshot(publication), publication);
    ASSERT_TRUE(held);
    ASSERT_TRUE(held->depth_cache_authorized());
    ASSERT_NE(held->depth_owner_frame_id, publication.frame_id);
    ASSERT_EQ(held->subtitle_frame_id, held->depth_owner_frame_id);
    for (unsigned idle = 0u; idle < 1000u; ++idle) {
      source.expect_idle();
    }
    EXPECT_EQ(source.status_reads, 0u);

    // New authority or explicit user work still wakes the exact same shared path.
    source.work.authority_reprocess_pending = true;
    source.expect_service();
    source.work.authority_reprocess_pending = false;
    source.work.dump_pending = true;
    source.expect_service();
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
