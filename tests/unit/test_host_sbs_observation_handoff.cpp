/**
 * @file tests/unit/test_host_sbs_observation_handoff.cpp
 * @brief CPU coverage of the production observation/submission composition boundary.
 */
#include <gtest/gtest.h>

#include "src/host_sbs_observation_handoff.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace {
  using handoff_t = host_sbs::observation_handoff_t;
  static_assert(std::is_same_v<decltype(std::declval<handoff_t &>().step()), const models::depth_step_result_t &>);

  models::depth_step_result_t step_for(const std::uint64_t frame, const std::uint64_t domain, const bool completed) {
    models::depth_step_result_t step;
    if (!completed) {
      return step;
    }
    auto &observation = step.observation;
    observation.raw_width = 1540;
    observation.raw_height = 868;
    observation.field_width = 1540;
    observation.field_height = 868;
    observation.field_content = {0u, 0u, 1540u, 868u};
    observation.input_region.source_width = 1920u;
    observation.input_region.source_height = 1080u;
    observation.input_region.right = 1920u;
    observation.input_region.bottom = 1080u;
    observation.input_region.analysis_generation = domain;
    observation.completed_frame_valid = completed;
    observation.completed_frame_id = completed ? frame : 0u;
    observation.publication.frame_id = frame;
    observation.publication.publication_sequence = frame + 100u;
    observation.publication.domain_tag = domain;
    observation.raw_model_provenance = std::make_shared<const models::raw_model_provenance_t>(
      models::raw_model_provenance_t {.depth_model = "owned test metadata"}
    );
    return step;
  }

  void expect_observation(const models::depth_observation_views_t &actual, const models::depth_observation_views_t &expected) {
    EXPECT_EQ(actual.completed_frame_valid, expected.completed_frame_valid);
    EXPECT_EQ(actual.completed_frame_id, expected.completed_frame_id);
    EXPECT_EQ(actual.publication, expected.publication);
    EXPECT_EQ(actual.raw_width, expected.raw_width);
    EXPECT_EQ(actual.raw_height, expected.raw_height);
    EXPECT_EQ(actual.field_width, expected.field_width);
    EXPECT_EQ(actual.field_height, expected.field_height);
    EXPECT_EQ(actual.field_content, expected.field_content);
    EXPECT_EQ(actual.input_region, expected.input_region);
    EXPECT_EQ(actual.raw_model_provenance, expected.raw_model_provenance);
  }

  void expect_no_submission(const models::depth_submission_t &submission) {
    EXPECT_FALSE(submission.inference_enqueued);
    EXPECT_FALSE(submission.gpu_undecided_transaction_enqueued);
    EXPECT_FALSE(submission.subtitle_ocr_inference_enqueued);
  }

  TEST(HostSbsObservationHandoffTest, SelectPreservesAdmissionAndStatusWithoutAnObservation) {
    handoff_t handoff;
    auto working = step_for(17u, 7u, false);
    working.submission.inference_enqueued = true;
    working.submission.subtitle_ocr_inference_enqueued = true;
    working.status.cuda_graph_active = true;
    handoff.select(working);

    const auto &selected = std::as_const(handoff).step();
    EXPECT_FALSE(selected.observation.completed_frame_valid);
    EXPECT_EQ(selected.observation.raw_width, 0);
    EXPECT_FALSE(selected.observation.raw_model_provenance);
    expect_observation(selected.observation, working.observation);
    EXPECT_TRUE(selected.submission.inference_enqueued);
    EXPECT_TRUE(selected.submission.subtitle_ocr_inference_enqueued);
    EXPECT_FALSE(selected.submission.gpu_undecided_transaction_enqueued);
    EXPECT_FALSE(selected.completion.input_domain_reset);
    EXPECT_TRUE(selected.status.cuda_graph_active);
  }

  TEST(HostSbsObservationHandoffTest, FinalizedCompletionKeepsItsViewsAndEventWhileSubmissionAdvances) {
    handoff_t handoff;
    auto completed_n = step_for(17u, 7u, true);
    completed_n.completion.input_domain_reset = true;
    completed_n.submission.gpu_undecided_transaction_enqueued = true;
    handoff.select(completed_n);

    auto submitted = step_for(18u, 7u, false);
    submitted.submission.inference_enqueued = true;
    submitted.submission.subtitle_ocr_inference_enqueued = true;
    ASSERT_TRUE(handoff.accept_submission(submitted, true));
    expect_observation(handoff.step().observation, completed_n.observation);
    EXPECT_TRUE(handoff.step().completion.input_domain_reset);
    EXPECT_TRUE(handoff.step().submission.inference_enqueued);
    EXPECT_TRUE(handoff.step().submission.subtitle_ocr_inference_enqueued);
    EXPECT_FALSE(handoff.step().submission.gpu_undecided_transaction_enqueued);

    auto next_submission = step_for(19u, 7u, false);
    next_submission.submission.gpu_undecided_transaction_enqueued = true;
    ASSERT_TRUE(handoff.accept_submission(next_submission, true));
    expect_observation(handoff.step().observation, completed_n.observation);
    EXPECT_TRUE(handoff.step().completion.input_domain_reset);
    EXPECT_FALSE(handoff.step().submission.inference_enqueued);
    EXPECT_FALSE(handoff.step().submission.subtitle_ocr_inference_enqueued);
    EXPECT_TRUE(handoff.step().submission.gpu_undecided_transaction_enqueued);
  }

  TEST(HostSbsObservationHandoffTest, RetainedCompletionCombinesOnlyCallStatus) {
    for (const bool earlier_status : {false, true}) {
      for (const bool later_status : {false, true}) {
        SCOPED_TRACE(earlier_status);
        SCOPED_TRACE(later_status);
        handoff_t handoff;
        auto selected = step_for(17u, 7u, true);
        selected.status.cuda_graph_active = earlier_status;
        handoff.select(selected);
        auto incoming = step_for(18u, 7u, false);
        incoming.status.cuda_graph_active = later_status;
        incoming.completion.input_domain_reset = true;

        ASSERT_TRUE(handoff.accept_submission(incoming, true));
        EXPECT_EQ(handoff.step().status.cuda_graph_active, earlier_status || later_status);
        EXPECT_FALSE(handoff.step().completion.input_domain_reset);
        expect_observation(handoff.step().observation, selected.observation);
      }
    }
  }

  TEST(HostSbsObservationHandoffTest, CallerRevocationReplacesTheWholeSelectedStep) {
    handoff_t handoff;
    auto earlier = step_for(17u, 7u, true);
    earlier.status.cuda_graph_active = true;
    earlier.completion.input_domain_reset = true;
    earlier.submission.gpu_undecided_transaction_enqueued = true;
    const auto metadata = std::weak_ptr {earlier.observation.raw_model_provenance};
    handoff.select(std::move(earlier));
    ASSERT_FALSE(metadata.expired());
    auto current = step_for(18u, 8u, false);
    current.submission.inference_enqueued = true;

    // The caller has already revoked its finalized verdict. This composer does not
    // authenticate domains or decide whether GPU resources remain current.
    ASSERT_FALSE(handoff.accept_submission(current, false));
    EXPECT_TRUE(metadata.expired());
    expect_observation(handoff.step().observation, current.observation);
    EXPECT_FALSE(handoff.step().observation.completed_frame_valid);
    EXPECT_FALSE(handoff.step().status.cuda_graph_active);
    EXPECT_FALSE(handoff.step().completion.input_domain_reset);
    EXPECT_TRUE(handoff.step().submission.inference_enqueued);
    EXPECT_FALSE(handoff.step().submission.gpu_undecided_transaction_enqueued);
    EXPECT_FALSE(handoff.step().submission.subtitle_ocr_inference_enqueued);
  }

  TEST(HostSbsObservationHandoffTest, IncomingCompletionReplacesEarlierCompletionAndStatus) {
    handoff_t handoff;
    auto earlier = step_for(17u, 7u, true);
    earlier.status.cuda_graph_active = true;
    earlier.completion.input_domain_reset = true;
    handoff.select(earlier);
    auto incoming = step_for(18u, 8u, true);
    incoming.submission.inference_enqueued = true;
    incoming.submission.subtitle_ocr_inference_enqueued = true;

    ASSERT_FALSE(handoff.accept_submission(incoming, true));
    expect_observation(handoff.step().observation, incoming.observation);
    EXPECT_FALSE(handoff.step().status.cuda_graph_active);
    EXPECT_FALSE(handoff.step().completion.input_domain_reset);
    EXPECT_TRUE(handoff.step().submission.inference_enqueued);
    EXPECT_TRUE(handoff.step().submission.subtitle_ocr_inference_enqueued);
    EXPECT_FALSE(handoff.step().submission.gpu_undecided_transaction_enqueued);
  }

  TEST(HostSbsObservationHandoffTest, FinalizedVerdictCannotPreserveAnIncompleteObservation) {
    handoff_t handoff;
    auto earlier = step_for(17u, 7u, false);
    earlier.status.cuda_graph_active = true;
    handoff.select(earlier);
    const auto incoming = step_for(18u, 8u, false);

    ASSERT_FALSE(handoff.accept_submission(incoming, true));
    expect_observation(handoff.step().observation, incoming.observation);
    EXPECT_FALSE(handoff.step().status.cuda_graph_active);
    EXPECT_FALSE(handoff.step().completion.input_domain_reset);
    expect_no_submission(handoff.step().submission);
  }

  TEST(HostSbsObservationHandoffTest, RepeatCopiesObservationAndStatusWithoutReplayingEventsOrSubmission) {
    auto source = step_for(17u, 7u, true);
    source.submission.inference_enqueued = true;
    source.submission.gpu_undecided_transaction_enqueued = true;
    source.submission.subtitle_ocr_inference_enqueued = true;
    source.completion.input_domain_reset = true;
    source.status.cuda_graph_active = true;
    const auto expected = source.observation;

    handoff_t original;
    original.select(std::move(source));
    handoff_t repeat;
    repeat.select_retained(original);
    original.reset();
    const auto &repeated = repeat.step();

    expect_observation(repeated.observation, expected);
    EXPECT_TRUE(repeated.status.cuda_graph_active);
    EXPECT_FALSE(repeated.completion.input_domain_reset);
    expect_no_submission(repeated.submission);
  }

  TEST(HostSbsObservationHandoffTest, ResetDropsEveryGroupAndOwnedObservationMetadata) {
    handoff_t handoff;
    auto selected = step_for(17u, 7u, true);
    selected.submission.inference_enqueued = true;
    selected.submission.gpu_undecided_transaction_enqueued = true;
    selected.submission.subtitle_ocr_inference_enqueued = true;
    selected.completion.input_domain_reset = true;
    selected.status.cuda_graph_active = true;
    const auto metadata = std::weak_ptr {selected.observation.raw_model_provenance};
    handoff.select(std::move(selected));
    ASSERT_FALSE(metadata.expired());

    handoff.reset();

    EXPECT_TRUE(metadata.expired());
    EXPECT_FALSE(handoff.step().observation.raw_model_provenance);
    EXPECT_FALSE(handoff.step().observation.completed_frame_valid);
    EXPECT_EQ(handoff.step().observation.completed_frame_id, 0u);
    EXPECT_EQ(handoff.step().observation.raw_width, 0);
    EXPECT_EQ(handoff.step().observation.publication, models::host_sbs_gpu_completion_receipt::expected_t {});
    EXPECT_FALSE(handoff.step().completion.input_domain_reset);
    EXPECT_FALSE(handoff.step().status.cuda_graph_active);
    expect_no_submission(handoff.step().submission);
  }
}  // namespace
