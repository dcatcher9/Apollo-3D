/**
 * @file tests/unit/test_sbs_roi_shape_transition.cpp
 * @brief Pure lifecycle tests for delayed active-ROI shape confirmation.
 */

#include <gtest/gtest.h>
#include <src/sbs_roi_shape_transition.h>

#include <utility>

namespace {
  using models::sbs_roi_shape_binding_failure;
  using models::sbs_roi_shape_binding_failure_action;
  using models::sbs_roi_canonical_recovery_requires_rebuild;
  using models::sbs_roi_shape_confirmation_guard;
  using models::sbs_roi_shape_confirmation_result;
  using models::sbs_roi_shape_sample_action;
  using models::sbs_roi_shape_sample_transition;
  using models::sbs_cuda_resource_cleanup_disposition;
  using models::sbs_cuda_resource_cleanup_policy;
  using models::sbs_trt_context_disposition;
  using models::sbs_trt_context_event;
  using models::sbs_trt_context_pool_accounting;
  using models::sbs_trt_context_accounting_after_quarantine;
  using models::sbs_trt_context_pool_disposition;
  using models::sbs_trt_context_reusable_after;

  TEST(SbsRoiShapeTransition, OlderRequestCannotConfirmNewerFrozenRuleState) {
    sbs_roi_shape_confirmation_guard guard;
    ASSERT_TRUE(guard.begin(102u, 3840u, 2160u, true));

    EXPECT_EQ(
      guard.observe(false, true, 101u, true),
      sbs_roi_shape_confirmation_result::waiting
    );
    EXPECT_TRUE(guard.awaiting());
    EXPECT_EQ(guard.source_frame_id(), 102u);

    EXPECT_EQ(
      guard.observe(false, true, 102u, true),
      sbs_roi_shape_confirmation_result::confirmed
    );
    EXPECT_FALSE(guard.awaiting());
  }

  TEST(SbsRoiShapeTransition, InvalidExactFrameDoesNotConfirm) {
    sbs_roi_shape_confirmation_guard guard;
    ASSERT_TRUE(guard.begin(77u, 1920u, 1080u, false));

    EXPECT_EQ(
      guard.observe(true, true, 77u, false),
      sbs_roi_shape_confirmation_result::waiting
    );
    EXPECT_TRUE(guard.copy_scheduled());
    EXPECT_TRUE(guard.awaiting());
  }

  TEST(SbsRoiShapeTransition, ConfirmationTimeoutRequestsCanonicalRecovery) {
    sbs_roi_shape_confirmation_guard guard;
    ASSERT_TRUE(guard.begin(7u, 3840u, 2160u, true));

    for (
      unsigned attempt = 1u;
      attempt <
        sbs_roi_shape_confirmation_guard::
          max_capture_opportunities;
      ++attempt
    ) {
      EXPECT_EQ(
        guard.observe(false, false, 0u, false),
        sbs_roi_shape_confirmation_result::waiting
      );
    }
    EXPECT_EQ(
      guard.observe(false, false, 0u, false),
      sbs_roi_shape_confirmation_result::recover_canonical
    );
    EXPECT_FALSE(guard.awaiting());
  }

  TEST(SbsRoiShapeTransition, InvalidBeginFailsClosed) {
    sbs_roi_shape_confirmation_guard guard;
    EXPECT_FALSE(guard.begin(0u, 3840u, 2160u, true));
    EXPECT_FALSE(guard.begin(1u, 0u, 2160u, true));
    EXPECT_FALSE(guard.awaiting());
  }

  TEST(SbsRoiShapeTransition, DynamicBindingFailureRecoversButCanonicalRetires) {
    EXPECT_EQ(
      sbs_roi_shape_binding_failure(
        true,
        560u,
        560u,
        770u,
        434u
      ),
      sbs_roi_shape_binding_failure_action::
        recover_canonical
    );
    EXPECT_EQ(
      sbs_roi_shape_binding_failure(
        true,
        770u,
        434u,
        770u,
        434u
      ),
      sbs_roi_shape_binding_failure_action::
        retire_estimator
    );
    EXPECT_EQ(
      sbs_roi_shape_binding_failure(
        false,
        560u,
        560u,
        770u,
        434u
      ),
      sbs_roi_shape_binding_failure_action::
        retire_estimator
    );
  }

  TEST(SbsRoiShapeTransition, SameShapeDelayedSampleDoesNotBubbleInference) {
    EXPECT_EQ(
      sbs_roi_shape_sample_transition(
        101u,
        102u,
        560u,
        560u,
        560u,
        560u
      ),
      sbs_roi_shape_sample_action::continue_current_shape
    );
  }

  TEST(SbsRoiShapeTransition, DestructiveShapeChangeRequiresExactFrame) {
    EXPECT_EQ(
      sbs_roi_shape_sample_transition(
        101u,
        102u,
        560u,
        560u,
        770u,
        434u
      ),
      sbs_roi_shape_sample_action::confirm_shape_change
    );
    EXPECT_EQ(
      sbs_roi_shape_sample_transition(
        102u,
        102u,
        560u,
        560u,
        770u,
        434u
      ),
      sbs_roi_shape_sample_action::apply_exact_shape_change
    );
  }

  TEST(SbsRoiShapeTransition, CanonicalRecoveryRebuildsEvenAtSameDimensions) {
    EXPECT_TRUE(sbs_roi_canonical_recovery_requires_rebuild(true, true));
    EXPECT_FALSE(sbs_roi_canonical_recovery_requires_rebuild(false, true));
    EXPECT_FALSE(sbs_roi_canonical_recovery_requires_rebuild(true, false));
  }

  TEST(SbsRoiShapeTransition, OnlyWarmedHealthyTensorRtContextsReturnToPool) {
    EXPECT_EQ(
      sbs_trt_context_pool_disposition(true, true),
      sbs_trt_context_disposition::reuse
    );
    EXPECT_EQ(
      sbs_trt_context_pool_disposition(false, true),
      sbs_trt_context_disposition::quarantine
    );
    EXPECT_EQ(
      sbs_trt_context_pool_disposition(true, false),
      sbs_trt_context_disposition::quarantine
    );
    EXPECT_EQ(
      sbs_trt_context_pool_disposition(false, false),
      sbs_trt_context_disposition::quarantine
    );
  }

  TEST(SbsRoiShapeTransition, CudaCleanupRequiresSelectedContextAndIdleStream) {
    EXPECT_EQ(
      sbs_cuda_resource_cleanup_policy(true, true),
      sbs_cuda_resource_cleanup_disposition::release
    );
    for (const auto &state : {
           std::pair {false, false},
           std::pair {false, true},
           std::pair {true, false},
         }) {
      EXPECT_EQ(
        sbs_cuda_resource_cleanup_policy(state.first, state.second),
        sbs_cuda_resource_cleanup_disposition::
          retain_until_process_exit
      );
    }
  }

  TEST(SbsRoiShapeTransition, EveryTerminalRuntimeEventPoisonsContext) {
    EXPECT_TRUE(sbs_trt_context_reusable_after(
      true,
      sbs_trt_context_event::none
    ));
    EXPECT_TRUE(sbs_trt_context_reusable_after(
      true,
      sbs_trt_context_event::recoverable_dynamic_shape_rejection
    ));
    for (const auto event : {
           sbs_trt_context_event::canonical_shape_rejection,
           sbs_trt_context_event::tensor_address_rejection,
           sbs_trt_context_event::enqueue_rejection,
           sbs_trt_context_event::cuda_context_failure,
           sbs_trt_context_event::cuda_stream_failure,
           sbs_trt_context_event::cuda_interop_failure,
         }) {
      EXPECT_FALSE(sbs_trt_context_reusable_after(true, event));
    }
    EXPECT_FALSE(sbs_trt_context_reusable_after(
      false,
      sbs_trt_context_event::none
    ));
  }

  TEST(SbsRoiShapeTransition, WarmedQuarantinePreservesPhysicalCount) {
    constexpr sbs_trt_context_pool_accounting before {
      2u,
      2u,
      0u,
    };
    constexpr auto after =
      sbs_trt_context_accounting_after_quarantine(before, true);
    EXPECT_EQ(after.usable, 1u);
    EXPECT_EQ(after.warmed, 1u);
    EXPECT_EQ(after.quarantined, 1u);
    EXPECT_EQ(
      before.usable + before.quarantined,
      after.usable + after.quarantined
    );

    constexpr auto cold =
      sbs_trt_context_accounting_after_quarantine(
        sbs_trt_context_pool_accounting {1u, 0u, 0u},
        false
      );
    EXPECT_EQ(cold.usable, 0u);
    EXPECT_EQ(cold.warmed, 0u);
    EXPECT_EQ(cold.quarantined, 1u);
  }
}  // namespace
