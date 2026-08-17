/**
 * @file src/host_sbs_gpu_trace.h
 * @brief Diagnostic-only GPU completion-ring wire contract for Dump 3D.
 */
#pragma once

#include "cuda_conditional_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace models::host_sbs_gpu_trace {
  inline constexpr std::uint32_t ring_schema = 3u;
  inline constexpr std::uint32_t dump_contract_schema = 3u;
  inline constexpr std::uint32_t decoded_trace_schema = 4u;
  inline constexpr std::uint32_t ring_tag = 0x48525447u;  // GTRH in little-endian memory.
  inline constexpr std::uint32_t record_tag = 0x31525447u;  // GTR1.
  inline constexpr std::uint32_t capacity = 300u;
  inline constexpr std::uint32_t header_word_count = 16u;
  inline constexpr std::uint32_t record_word_count = 176u;
  inline constexpr std::uint32_t constant_word_count = 20u;
  inline constexpr std::uint32_t transaction_word_count = 64u;
  inline constexpr std::uint32_t subtitle_locator_word_count = 80u;
  inline constexpr std::uint32_t subtitle_condition_word_count = 6u;
  inline constexpr std::uint32_t ring_word_count =
    header_word_count + capacity * record_word_count;
  inline constexpr std::uint32_t ring_byte_count =
    ring_word_count * sizeof(std::uint32_t);

  enum class header_word_e : std::uint32_t {
    schema = 0u,
    tag = 1u,
    capacity = 2u,
    record_words = 3u,
    next_sequence_low = 4u,
    next_sequence_high = 5u,
    next_slot = 6u,
    committed_count = 7u,
    reserved_begin = 8u,
  };

  enum class record_word_e : std::uint32_t {
    schema = 0u,
    commit_tag = 1u,
    sequence_low = 2u,
    sequence_high = 3u,
    frame_low = 4u,
    frame_high = 5u,
    analysis_generation_low = 6u,
    analysis_generation_high = 7u,
    domain_tag_low = 8u,
    domain_tag_high = 9u,
    transaction_token_low = 10u,
    transaction_token_high = 11u,
    submission_class = 12u,
    depth_disposition = 13u,
    expected_work = 14u,
    subtitle_disposition = 15u,
    flags = 16u,
    host_subtitle_outcome = 17u,
    source_width = 18u,
    source_height = 19u,
    field_width = 20u,
    field_height = 21u,
    transaction_words = 22u,
    reserved0 = 23u,
    transaction_begin = 24u,
    subtitle_locator_begin = transaction_begin + transaction_word_count,
    subtitle_condition_begin = subtitle_locator_begin + subtitle_locator_word_count,
    observation_timestamp_low = subtitle_condition_begin + subtitle_condition_word_count,
    observation_timestamp_high = observation_timestamp_low + 1u,
    reserved_begin = observation_timestamp_high + 1u,
  };

  enum class submission_class_e : std::uint32_t {
    invalid = 0u,
    force_infer = 1u,
    gpu_undecided = 2u,
  };

  enum class depth_disposition_e : std::uint32_t {
    invalid = 0u,
    reuse = 1u,
    infer = 2u,
  };

  enum class subtitle_disposition_e : std::uint32_t {
    suppressed = 0u,
    optional_ocr = 1u,
    abstention = 2u,
    held_with_depth = 5u,
    invalid = 6u,
  };

  enum class host_subtitle_outcome_e : std::uint32_t {
    suppressed = 0u,
    ordinary_record = 1u,
  };

  enum record_flag_e : std::uint32_t {
    input_domain_reset = 1u << 0u,
    dump_forced = 1u << 1u,
    ocr_record_submitted = 1u << 2u,
    subtitle_suppressed = 1u << 3u,
    condition_executed = 1u << 4u,
    subtitle_branch_gated = 1u << 5u,
  };
  inline constexpr std::uint32_t known_record_flags =
    input_domain_reset | dump_forced | ocr_record_submitted | subtitle_suppressed |
    condition_executed | subtitle_branch_gated;

  struct receipt_authentication_t {
    depth_disposition_e depth = depth_disposition_e::invalid;
    bool receipt_valid = false;
    bool optional_ocr_executed = false;
  };

  [[nodiscard]] constexpr std::size_t word_index(
    const header_word_e word
  ) noexcept {
    return static_cast<std::size_t>(word);
  }

  [[nodiscard]] constexpr std::size_t word_index(
    const record_word_e word
  ) noexcept {
    return static_cast<std::size_t>(word);
  }

  [[nodiscard]] constexpr std::size_t record_base(
    const std::uint32_t slot
  ) noexcept {
    return header_word_count + static_cast<std::size_t>(slot) * record_word_count;
  }

  [[nodiscard]] constexpr std::uint64_t join_u64(
    const std::uint32_t low,
    const std::uint32_t high
  ) noexcept {
    return static_cast<std::uint64_t>(low) |
           (static_cast<std::uint64_t>(high) << 32u);
  }

  [[nodiscard]] constexpr cuda_conditional_graph::decision_record_t
  decision_record_from_transaction(
    const std::array<std::uint32_t, transaction_word_count> &words
  ) noexcept {
    return {
      words[0u], words[1u], words[2u], words[3u],
      words[4u], words[5u], words[6u], words[7u],
    };
  }

  [[nodiscard]] constexpr cuda_conditional_graph::request_record_t
  request_record_from_transaction(
    const std::array<std::uint32_t, transaction_word_count> &words
  ) noexcept {
    return {
      words[8u], words[9u], words[10u], words[11u],
      words[12u], words[13u], words[14u], words[15u],
    };
  }

  [[nodiscard]] constexpr receipt_authentication_t authenticate_receipt(
    const std::array<std::uint32_t, transaction_word_count> &words,
    const std::uint64_t expected_token,
    const std::uint32_t expected_work,
    const submission_class_e submission_class
  ) noexcept {
    const auto request = request_record_from_transaction(words);
    const auto receipt = decision_record_from_transaction(words);
    const bool expected_valid = expected_token != 0u &&
      cuda_conditional_graph::authenticated_work_flags(expected_work);
    const bool request_matches = expected_valid &&
      cuda_conditional_graph::authenticated_request(request) &&
      join_u64(request.token_low, request.token_high) == expected_token &&
      request.work_flags == expected_work;
    if (!request_matches ||
        !cuda_conditional_graph::authenticated_receipt(receipt, request)) {
      return {};
    }
    const auto branch = static_cast<cuda_conditional_graph::branch_e>(receipt.decision);
    if (submission_class != submission_class_e::force_infer &&
        submission_class != submission_class_e::gpu_undecided) {
      return {};
    }
    if (submission_class == submission_class_e::force_infer &&
        branch != cuda_conditional_graph::branch_e::infer) {
      return {};
    }
    return {
      branch == cuda_conditional_graph::branch_e::reuse ?
        depth_disposition_e::reuse : depth_disposition_e::infer,
      true,
      cuda_conditional_graph::authenticated_optional_ocr_receipt(receipt, request),
    };
  }

  [[nodiscard]] constexpr subtitle_disposition_e classify_subtitle_disposition(
    const std::uint32_t expected_work,
    const host_subtitle_outcome_e host_outcome,
    const receipt_authentication_t &receipt,
    const std::uint32_t flags
  ) noexcept {
    using cuda_conditional_graph::work_flag_e;
    using cuda_conditional_graph::work_flags_value;
    if ((flags & ~known_record_flags) != 0u) {
      return subtitle_disposition_e::invalid;
    }
    const bool record_published = (flags & ocr_record_submitted) != 0u;
    const bool suppressed = (flags & subtitle_suppressed) != 0u;
    const bool conditioned = (flags & condition_executed) != 0u;
    const bool branch_gated = (flags & subtitle_branch_gated) != 0u;
    if (branch_gated) {
      // Host flags describe only work the CPU can prove. For an opaque transaction, the same
      // authenticated receipt that gates the indirect dispatches supplies execution authority.
      if (!receipt.receipt_valid || expected_work == work_flags_value(work_flag_e::none) ||
          suppressed || record_published || conditioned) {
        return subtitle_disposition_e::invalid;
      }
      const bool infer_coupled =
        (expected_work == work_flags_value(work_flag_e::optional_ocr) ||
         expected_work == work_flags_value(work_flag_e::subtitle_observation)) &&
        host_outcome == host_subtitle_outcome_e::ordinary_record;
      const bool cadence_due =
        (expected_work == work_flags_value(work_flag_e::optional_ocr_due) ||
         expected_work == work_flags_value(work_flag_e::subtitle_observation_due)) &&
        host_outcome == host_subtitle_outcome_e::ordinary_record;
      if (!infer_coupled && !cadence_due) {
        return subtitle_disposition_e::invalid;
      }
      if (infer_coupled && receipt.depth == depth_disposition_e::reuse) {
        return !receipt.optional_ocr_executed ?
          subtitle_disposition_e::held_with_depth : subtitle_disposition_e::invalid;
      }
      if (receipt.depth != depth_disposition_e::infer &&
          !(cadence_due && receipt.depth == depth_disposition_e::reuse)) {
        return subtitle_disposition_e::invalid;
      }
      if (expected_work == work_flags_value(work_flag_e::optional_ocr) || cadence_due) {
        return receipt.optional_ocr_executed ?
          subtitle_disposition_e::optional_ocr : subtitle_disposition_e::abstention;
      }
      if (expected_work == work_flags_value(work_flag_e::subtitle_observation)) {
        return subtitle_disposition_e::abstention;
      }
      return subtitle_disposition_e::invalid;
    }
    if (expected_work == work_flags_value(work_flag_e::none)) {
      return host_outcome == host_subtitle_outcome_e::suppressed && suppressed &&
          !record_published && !conditioned ?
        subtitle_disposition_e::suppressed : subtitle_disposition_e::invalid;
    }
    if (expected_work == work_flags_value(work_flag_e::optional_ocr) ||
        expected_work == work_flags_value(work_flag_e::optional_ocr_due) ||
        expected_work == work_flags_value(work_flag_e::subtitle_observation_due)) {
      if (host_outcome != host_subtitle_outcome_e::ordinary_record || suppressed ||
          !record_published || !conditioned) {
        return subtitle_disposition_e::invalid;
      }
      return receipt.optional_ocr_executed ?
        subtitle_disposition_e::optional_ocr : subtitle_disposition_e::abstention;
    }
    if (expected_work == work_flags_value(work_flag_e::subtitle_observation)) {
      return host_outcome == host_subtitle_outcome_e::ordinary_record && !suppressed &&
          record_published && conditioned ?
        subtitle_disposition_e::abstention : subtitle_disposition_e::invalid;
    }
    return subtitle_disposition_e::invalid;
  }

  static_assert(record_word_count * sizeof(std::uint32_t) <= 1024u);
  static_assert(word_index(record_word_e::transaction_begin) + transaction_word_count ==
                word_index(record_word_e::subtitle_locator_begin));
  static_assert(word_index(record_word_e::subtitle_locator_begin) +
                  subtitle_locator_word_count ==
                word_index(record_word_e::subtitle_condition_begin));
  static_assert(word_index(record_word_e::subtitle_condition_begin) +
                  subtitle_condition_word_count ==
                word_index(record_word_e::observation_timestamp_low));
  static_assert(word_index(record_word_e::observation_timestamp_low) == 174u);
  static_assert(word_index(record_word_e::observation_timestamp_high) == 175u);
  static_assert(word_index(record_word_e::reserved_begin) == record_word_count);
  static_assert(ring_word_count == 52816u);
  static_assert(ring_byte_count == 211264u);
}  // namespace models::host_sbs_gpu_trace
