#include <gtest/gtest.h>

#include <src/host_sbs_adaptive_submission.h>

namespace {
  namespace receipt = models::host_sbs_gpu_completion_receipt;

  receipt::receipt_t completed_reuse() {
    return {
      .expected = {
        .estimator_generation = 3u,
        .publication_sequence = 8u,
        .analysis_generation = 5u,
        .frame_id = 42u,
        .transaction_token = 42u,
        .domain_tag = 9u,
        .observation_timestamp_us = 2000u,
        .baseline_frame_id = 40u,
        .width = 320u,
        .height = 180u,
        .raw_coordinate_scale = 1.0f,
        .expected_work = 2u,
        .submission_class = receipt::submission_class_e::gpu_undecided,
      },
      .depth = receipt::depth_disposition_e::reuse,
      .geometry_valid = true,
      .depth_owner_valid = true,
      .depth_owner_frame_id = 40u,
      .depth_owner_timestamp_us = 1000u,
      .subtitle = receipt::subtitle_disposition_e::held_with_depth,
      .subtitle_frame_id = 40u,
      .subtitle_domain_tag = 9u,
    };
  }

  // Models the caller's mutable resource publication separately from a copied result/receipt.
  // The same frame id may be submitted again; it cannot authenticate overwritten field aliases.
  struct publication_adapter_t {
    receipt::expected_t current;
    std::optional<receipt::receipt_t> observed;

    void poll(const receipt::receipt_t &queued, bool event_ready) {
      if (event_ready && receipt::matches_publication(queued, current)) {
        observed = queued;
      }
    }

    models::gpu_adaptive_reuse_request admit() const {
      return admit(observed ? &*observed : nullptr);
    }

    models::gpu_adaptive_reuse_request admit(const receipt::receipt_t *completed) const {
      return models::make_gpu_adaptive_request(43u, true, completed, current, 3000u);
    }
  };

  TEST(HostSbsAdaptiveSubmissionTest, ReuseKeepsOneAnalysisOwnerApartFromCurrentColor) {
    const auto completed = completed_reuse();
    const publication_adapter_t adapter {completed.expected};
    const auto request = adapter.admit(&completed);
    ASSERT_TRUE(request.authorize_gpu_undecided_reuse);
    EXPECT_EQ(request.baseline_frame_id, 40u);
    EXPECT_EQ(request.gpu_reuse_decision_token, 43u);
    EXPECT_EQ(completed.expected.frame_id, 42u);
    EXPECT_EQ(completed.depth_owner_timestamp_us, 1000u);
    EXPECT_FALSE(completed.subtitle_is_current());
    EXPECT_EQ(completed.subtitle_frame_id, completed.depth_owner_frame_id);
  }



  TEST(HostSbsAdaptiveSubmissionTest, LateEventCanAuthorizeCurrentPublicationWithoutWaiting) {
    const auto completed = completed_reuse();
    publication_adapter_t adapter {completed.expected};
    adapter.poll(completed, false);  // The early nonblocking query was not ready.
    EXPECT_FALSE(adapter.admit().authorize_gpu_undecided_reuse);
    adapter.poll(completed, true);   // Normal route/copy work gave the GPU time to finish.
    const auto late = adapter.admit();
    ASSERT_TRUE(late.authorize_gpu_undecided_reuse);
    EXPECT_EQ(late.baseline_frame_id, 40u);
    EXPECT_EQ(late.gpu_reuse_decision_token, 43u);
    EXPECT_EQ(adapter.observed->expected.frame_id, 42u);
  }

  TEST(HostSbsAdaptiveSubmissionTest, LateBusyOrOverwrittenEventStillUsesForceFallback) {
    const auto completed = completed_reuse();
    publication_adapter_t adapter {completed.expected};
    adapter.poll(completed, false);
    adapter.poll(completed, false);
    EXPECT_FALSE(adapter.admit().authorize_gpu_undecided_reuse);
    ++adapter.current.publication_sequence;
    adapter.poll(completed, true);   // A ready older event cannot describe overwritten fields.
    EXPECT_FALSE(adapter.observed.has_value());
    EXPECT_FALSE(adapter.admit().authorize_gpu_undecided_reuse);
  }

  TEST(HostSbsAdaptiveSubmissionTest, LateReceiptCannotAuthorizeOverwrittenBorrowedFields) {
    const auto completed = completed_reuse();
    publication_adapter_t adapter {completed.expected};
    ASSERT_TRUE(adapter.admit(&completed).authorize_gpu_undecided_reuse);
    ++adapter.current.publication_sequence;  // Same color frame, different field publication.
    const auto rejected = adapter.admit(&completed);
    EXPECT_FALSE(rejected.authorize_gpu_undecided_reuse);
    EXPECT_EQ(rejected.baseline_frame_id, 0u);
    EXPECT_EQ(rejected.observation_timestamp_us, 3000u);
    adapter.current = completed.expected;
    ++adapter.current.estimator_generation;
    EXPECT_FALSE(adapter.admit(&completed).authorize_gpu_undecided_reuse);
  }

  TEST(HostSbsAdaptiveSubmissionTest, PendingMissingInvalidAndOldDomainUseForcePath) {
    auto completed = completed_reuse();
    publication_adapter_t adapter {completed.expected};
    EXPECT_FALSE(adapter.admit(nullptr).authorize_gpu_undecided_reuse);
    completed.geometry_valid = false;
    EXPECT_FALSE(adapter.admit(&completed).authorize_gpu_undecided_reuse);
    completed.geometry_valid = true;
    completed.depth_owner_valid = false;
    EXPECT_FALSE(adapter.admit(&completed).authorize_gpu_undecided_reuse);
    completed.depth_owner_valid = true;
    ++adapter.current.domain_tag;
    EXPECT_FALSE(adapter.admit(&completed).authorize_gpu_undecided_reuse);
  }

  TEST(HostSbsAdaptiveSubmissionTest, CurrentConditionalInferBecomesTheRealOwner) {
    auto completed = completed_reuse();
    completed.depth = receipt::depth_disposition_e::infer;
    completed.depth_owner_frame_id = completed.expected.frame_id;
    completed.depth_owner_timestamp_us = completed.expected.observation_timestamp_us;
    completed.subtitle = receipt::subtitle_disposition_e::optional_ocr;
    completed.subtitle_frame_id = completed.expected.frame_id;
    const publication_adapter_t adapter {completed.expected};
    const auto request = adapter.admit(&completed);
    ASSERT_TRUE(request.authorize_gpu_undecided_reuse);
    EXPECT_EQ(request.baseline_frame_id, 42u);
  }

  TEST(HostSbsAdaptiveSubmissionTest, HostProofAndObservationOrderStillConstrainAdmission) {
    const auto completed = completed_reuse();
    EXPECT_FALSE(models::make_gpu_adaptive_request(43u, false, &completed, completed.expected, 3000u)
                   .authorize_gpu_undecided_reuse);
    EXPECT_FALSE(models::make_gpu_adaptive_request(42u, true, &completed, completed.expected, 3000u)
                   .authorize_gpu_undecided_reuse);
    EXPECT_FALSE(models::make_gpu_adaptive_request(43u, true, &completed, completed.expected, 1999u)
                   .authorize_gpu_undecided_reuse);
    EXPECT_FALSE(models::make_gpu_adaptive_request(43u, true, &completed, completed.expected, 0u)
                   .authorize_gpu_undecided_reuse);
  }

  TEST(HostSbsAdaptiveSubmissionTest, ClassificationAcceptsDeviceFallbackWithoutInventingCompletion) {
    const auto completed = completed_reuse();
    const auto request = publication_adapter_t {completed.expected}.admit(&completed);
    using classification = models::gpu_adaptive_submission_class_e;
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, request, false, true), classification::gpu_undecided);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, request, true, false), classification::force_infer);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, request, true, true), classification::invalid);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, request, false, false), classification::invalid);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(44u, request, false, true), classification::invalid);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, {}, false, true), classification::invalid);
  }

  TEST(HostSbsAdaptiveSubmissionTest, OldAnalysisAndManyDirtyPublicationsRemainDetectorCandidates) {
    auto completed = completed_reuse();
    for (std::uint64_t dirty = 1u; dirty <= 1000u; ++dirty) {
      const auto frame = completed.expected.frame_id + 1u;
      const auto timestamp = completed.expected.observation_timestamp_us + 3600000000u;
      const auto request = models::make_gpu_adaptive_request(
        frame, true, &completed, completed.expected, timestamp
      );
      ASSERT_TRUE(request.authorize_gpu_undecided_reuse);
      EXPECT_EQ(request.baseline_frame_id, 40u);
      EXPECT_EQ(models::classify_gpu_adaptive_submission(frame, request, false, true),
                models::gpu_adaptive_submission_class_e::gpu_undecided);
      // Model another authenticated near-reuse completion: color advances, while the full
      // depth/subtitle tuple still belongs to the same original analyzed image.
      completed.expected.frame_id = frame;
      completed.expected.transaction_token = frame;
      ++completed.expected.publication_sequence;
      completed.expected.observation_timestamp_us = timestamp;
    }
    EXPECT_EQ(completed.depth_owner_frame_id, 40u);
    EXPECT_EQ(completed.subtitle_frame_id, 40u);
    EXPECT_EQ(completed.depth_owner_timestamp_us, 1000u);
  }

  TEST(HostSbsAdaptiveSubmissionTest, SuppressedSubtitlePublicationCannotSeedOrdinaryJointReuse) {
    auto completed = completed_reuse();
    completed.expected.submission_class = receipt::submission_class_e::force_infer;
    completed.expected.baseline_frame_id = 0u;
    completed.expected.expected_work = 0u;
    completed.expected.flags |= models::host_sbs_gpu_trace::subtitle_suppressed;
    completed.expected.host_subtitle_outcome = models::host_sbs_gpu_trace::host_subtitle_outcome_e::suppressed;
    completed.depth = receipt::depth_disposition_e::infer;
    completed.depth_owner_frame_id = completed.expected.frame_id;
    completed.depth_owner_timestamp_us = completed.expected.observation_timestamp_us;
    completed.subtitle = receipt::subtitle_disposition_e::suppressed;
    const auto request = publication_adapter_t {completed.expected}.admit(&completed);
    EXPECT_FALSE(request.authorize_gpu_undecided_reuse);
    EXPECT_EQ(request.baseline_frame_id, 0u);
    EXPECT_EQ(models::classify_gpu_adaptive_submission(43u, request, true, false),
              models::gpu_adaptive_submission_class_e::force_infer);
  }

  TEST(HostSbsAdaptiveSubmissionTest, ExplicitSubtitleSuppressionStillControlsEligibility) {
    using work = models::depth_optional_work_mode_e;
    EXPECT_EQ(models::select_depth_optional_work_mode(false, false), work::ordinary);
    EXPECT_EQ(models::select_depth_optional_work_mode(true, false), work::suppress_subtitle);
    EXPECT_EQ(models::select_depth_optional_work_mode(true, true), work::ordinary);
    EXPECT_TRUE(models::depth_optional_work_allows_gpu_undecided(work::ordinary));
    EXPECT_FALSE(models::depth_optional_work_allows_gpu_undecided(work::suppress_subtitle));
  }
}  // namespace
