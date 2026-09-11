#pragma once

#include <src/host_sbs_gpu_completion_receipt.h>

namespace host_sbs_gpu_completion_receipt_fixture {
  namespace receipt = models::host_sbs_gpu_completion_receipt;
  namespace graph = cuda_conditional_graph;
  namespace trace = models::host_sbs_gpu_trace;
  namespace v2 = models::depth_coordinate_v2;

  inline receipt::expected_t expected(
    graph::work_flag_e work = graph::work_flag_e::optional_ocr,
    receipt::submission_class_e submission = receipt::submission_class_e::gpu_undecided
  ) {
    const bool suppressed = work == graph::work_flag_e::none;
    return {
      .estimator_generation = 7u,
      .publication_sequence = 3u,
      .analysis_generation = 9u,
      .frame_id = 12u,
      .transaction_token = 0x123456780000000Cu,
      .domain_tag = 0xAABBCCDD00112233u,
      .observation_timestamp_us = 12000u,
      .baseline_frame_id = submission == receipt::submission_class_e::gpu_undecided ? 11u : 0u,
      .width = 770u,
      .height = 434u,
      .raw_coordinate_scale = 4.0f,
      .expected_work = graph::work_flags_value(work),
      .submission_class = submission,
      .flags = suppressed ? trace::subtitle_suppressed : submission == receipt::submission_class_e::gpu_undecided ?
                                                          trace::subtitle_branch_gated :
                                                          trace::ocr_record_submitted | trace::condition_executed,
      .host_subtitle_outcome = suppressed ? trace::host_subtitle_outcome_e::suppressed : trace::host_subtitle_outcome_e::ordinary_record,
    };
  }

  inline void put_u64(receipt::snapshot_words_t &words, std::size_t offset, std::uint64_t value) {
    words[offset] = static_cast<std::uint32_t>(value);
    words[offset + 1u] = static_cast<std::uint32_t>(value >> 32u);
  }

  inline receipt::snapshot_words_t snapshot(
    const receipt::expected_t &expected,
    graph::branch_e branch = graph::branch_e::reuse,
    bool optional_child_present = true
  ) {
    receipt::snapshot_words_t words {};
    const auto request = graph::make_request(expected.transaction_token, static_cast<graph::work_flag_e>(expected.expected_work));
    const auto decision = graph::resolve_proposal(graph::make_proposal(branch, expected.transaction_token), request, optional_child_present);
    const auto decision_words = std::bit_cast<std::array<std::uint32_t, 8u>>(decision);
    const auto request_words = std::bit_cast<std::array<std::uint32_t, 8u>>(request);
    std::copy(decision_words.begin(), decision_words.end(), words.begin());
    std::copy(request_words.begin(), request_words.end(), words.begin() + 8u);
    const auto dispatch = [&](std::size_t offset, std::uint32_t x, std::uint32_t y = 1u) {
      words[offset] = x;
      words[offset + 1u] = y;
      words[offset + 2u] = 1u;
    };
    const auto grid_x = (expected.width + 15u) / 16u;
    const auto grid_y = (expected.height + 15u) / 16u;
    const bool infer = branch == graph::branch_e::infer;
    if (expected.submission_class == receipt::submission_class_e::gpu_undecided) {
      dispatch(16u, infer ? static_cast<std::uint32_t>(std::min<std::uint64_t>(64u, (static_cast<std::uint64_t>(expected.width) * expected.height + 255u) / 256u)) : 0u);
      dispatch(20u, infer ? 1u : 0u);
      dispatch(24u, infer ? grid_x : 0u, infer ? grid_y : 1u);
      dispatch(32u, infer ? expected.width : 0u);
      dispatch(36u, infer ? expected.height : 0u);
      dispatch(40u, infer ? 0u : grid_x, infer ? 1u : grid_y);
    }
    const auto authentication = trace::authenticate_receipt(
      [&]() {
        std::array<std::uint32_t, trace::transaction_word_count> transaction {};
        std::copy_n(words.begin(), transaction.size(), transaction.begin());
        return transaction;
      }(),
      expected.transaction_token, expected.expected_work, expected.submission_class
    );
    const auto subtitle = trace::classify_subtitle_disposition(expected.expected_work, expected.host_subtitle_outcome, authentication, expected.flags);
    const bool observation = subtitle == receipt::subtitle_disposition_e::optional_ocr || subtitle == receipt::subtitle_disposition_e::abstention;
    const bool optional = subtitle == receipt::subtitle_disposition_e::optional_ocr;
    dispatch(44u, observation ? grid_x : 0u, observation ? grid_y : 1u);
    dispatch(48u, optional ? (v2::subtitle_ocr_output_width / 8u + 31u) / 32u : 0u,
      optional ? (v2::subtitle_ocr_output_height + 3u) / 4u : 1u);
    dispatch(52u, optional ? 1u : 0u);
    dispatch(56u, subtitle == receipt::subtitle_disposition_e::abstention ? 1u : 0u);
    dispatch(60u, observation ? 1u : 0u);

    const auto owner = receipt::history_owner_begin;
    words[owner] = receipt::history_owner_tag;
    words[owner + 1u] = receipt::history_owner_schema;
    put_u64(words, owner + 2u, infer ? expected.frame_id : expected.baseline_frame_id);
    put_u64(words, owner + 4u, expected.domain_tag);
    words[owner + 6u] = expected.width;
    words[owner + 7u] = expected.height;
    put_u64(words, owner + 8u, infer ? expected.observation_timestamp_us : expected.observation_timestamp_us - 1000u);
    if (subtitle != receipt::subtitle_disposition_e::suppressed) {
      const auto locator = receipt::subtitle_locator_begin;
      words[locator] = v2::subtitle_locator_state_schema;
      words[locator + 1u] = v2::subtitle_locator_state_tag;
      put_u64(words, locator + 10u, expected.analysis_generation);
      put_u64(words, locator + 22u, observation ? expected.frame_id : expected.baseline_frame_id);
      words[locator + 27u] = expected.width;
      words[locator + 28u] = expected.height;
    }
    auto state = v2::state_initial_words;
    state[v2::inverse_scale] = std::bit_cast<std::uint32_t>(1.0f / expected.raw_coordinate_scale);
    state[v2::calibration_revision] = 1u;
    state[v2::frame_valid] = std::bit_cast<std::uint32_t>(1.0f);
    state[v2::renderer_authorization_bits] = v2::contract_tag;
    state[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_words(
      state[v2::center], state[v2::inverse_scale], state[v2::convergence_curve], state[v2::calibration_revision]
    );
    std::copy(state.begin(), state.end(), words.begin() + receipt::parallax_state_begin);
    words[receipt::depth_frame_state_begin + 1u] = std::bit_cast<std::uint32_t>(1.0f);
    words[receipt::depth_frame_state_begin + 2u] = std::bit_cast<std::uint32_t>(1.0f);
    words[receipt::depth_frame_state_begin + 3u] = std::bit_cast<std::uint32_t>(1.0f);
    std::copy(sbs_adaptive_state::initial_words.begin(), sbs_adaptive_state::initial_words.end(),
      words.begin() + receipt::cut_state_begin);
    std::copy(models::host_sbs_gpu_outcomes::initial_words.begin(), models::host_sbs_gpu_outcomes::initial_words.end(),
      words.begin() + receipt::outcome_counts_begin);
    put_u64(words, receipt::outcome_counts_begin + 2u, 10u);
    put_u64(words, receipt::outcome_counts_begin + 4u, 7u);
    put_u64(words, receipt::outcome_counts_begin + 6u, 2u);
    return words;
  }
}  // namespace host_sbs_gpu_completion_receipt_fixture
