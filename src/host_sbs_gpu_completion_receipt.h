/**
 * @file src/host_sbs_gpu_completion_receipt.h
 * @brief CPU authentication of an ordered, completed Host SBS publication snapshot.
 */
#pragma once

#include "depth_coordinate_v2.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "host_sbs_gpu_outcomes.h"
#include "host_sbs_gpu_trace.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace models::host_sbs_gpu_completion_receipt {
  namespace trace = host_sbs_gpu_trace;
  namespace v2 = depth_coordinate_v2;
  using depth_disposition_e = trace::depth_disposition_e;
  using subtitle_disposition_e = trace::subtitle_disposition_e;
  using submission_class_e = trace::submission_class_e;
  using host_subtitle_outcome_e = trace::host_subtitle_outcome_e;

  // Copy these existing resources in order after the complete postprocess/conditioner unit,
  // then End one D3D event. A signalled event and DO_NOT_WAIT Map authorize decode; neither a
  // CUDA event nor a tag store alone proves that the complete field has been published.
  inline constexpr std::size_t transaction_begin = 0u;
  inline constexpr std::size_t transaction_word_count = 64u;
  inline constexpr std::size_t history_owner_begin = 64u;
  inline constexpr std::size_t history_owner_word_count = 10u;
  inline constexpr std::size_t subtitle_locator_begin = 74u;
  inline constexpr std::size_t subtitle_locator_word_count = 32u;
  inline constexpr std::size_t subtitle_condition_begin = 106u;
  inline constexpr std::size_t subtitle_condition_word_count = 6u;
  inline constexpr std::size_t parallax_state_begin = 112u;
  inline constexpr std::size_t parallax_state_word_count = 12u;
  inline constexpr std::size_t depth_frame_state_begin = 124u;
  inline constexpr std::size_t depth_frame_state_word_count = 4u;
  inline constexpr std::size_t cut_state_begin = 128u;
  inline constexpr std::size_t cut_state_word_count = sbs_adaptive_state::word_count;
  inline constexpr std::size_t outcome_counts_begin = cut_state_begin + cut_state_word_count;
  inline constexpr std::size_t outcome_counts_word_count = host_sbs_gpu_outcomes::word_count;
  inline constexpr std::size_t snapshot_word_count = outcome_counts_begin + outcome_counts_word_count;
  inline constexpr std::size_t snapshot_byte_count = snapshot_word_count * sizeof(std::uint32_t);
  inline constexpr std::size_t slot_count = 3u;
  using snapshot_words_t = std::array<std::uint32_t, snapshot_word_count>;

  // Existing host_sbs_near_identical_history_owner.hlsl wire ABI, not a new GPU algorithm.
  inline constexpr std::uint32_t history_owner_tag = 0x3142484Eu;
  inline constexpr std::uint32_t history_owner_schema = 2u;

  struct expected_t {
    std::uint64_t estimator_generation = 0u;
    std::uint64_t publication_sequence = 0u;
    std::uint64_t analysis_generation = 0u;
    std::uint64_t frame_id = 0u;
    std::uint64_t transaction_token = 0u;
    std::uint64_t domain_tag = 0u;
    std::uint64_t observation_timestamp_us = 0u;
    std::uint64_t baseline_frame_id = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    float raw_coordinate_scale = 0.0f;
    std::uint32_t expected_work = 0u;
    submission_class_e submission_class = submission_class_e::invalid;
    std::uint32_t flags = 0u;
    host_subtitle_outcome_e host_subtitle_outcome = host_subtitle_outcome_e::suppressed;

    bool operator==(const expected_t &) const = default;
  };

  struct receipt_t {
    expected_t expected;
    depth_disposition_e depth = depth_disposition_e::invalid;
    bool geometry_valid = false;
    bool depth_owner_valid = false;
    std::uint64_t depth_owner_frame_id = 0u;
    std::uint64_t depth_owner_timestamp_us = 0u;
    subtitle_disposition_e subtitle = subtitle_disposition_e::invalid;
    std::uint64_t subtitle_frame_id = 0u;
    std::uint64_t subtitle_domain_tag = 0u;

    // A selected infer can complete without advancing valid comparison/depth history.
    [[nodiscard]] bool depth_cache_authorized() const noexcept {
      return geometry_valid && depth_owner_valid;
    }
    [[nodiscard]] bool subtitle_is_current() const noexcept {
      return (subtitle == subtitle_disposition_e::optional_ocr || subtitle == subtitle_disposition_e::abstention) &&
             subtitle_frame_id == expected.frame_id && subtitle_domain_tag == expected.domain_tag;
    }
  };

  // Optional diagnostic payloads share the publication's copy/event/map, but never its rendering
  // authority. Decode them independently so bad geometry cannot hide cumulative failure counts
  // and a malformed diagnostic payload cannot revoke an otherwise valid publication.
  struct diagnostic_sample_t {
    expected_t expected;
    std::chrono::steady_clock::time_point sampled_at {};
    std::optional<sbs_adaptive_state::words_t> cut_state;
    std::optional<host_sbs_gpu_outcomes::counts_t> outcome_counts;
    bool failed = false;
  };

  [[nodiscard]] inline bool valid_cut_state_words(const sbs_adaptive_state::words_t &words) noexcept {
    using sbs_adaptive_state::word_e;
    const auto scalar = [&](const word_e word) {
      return std::bit_cast<float>(words[sbs_adaptive_state::index(word)]);
    };
    if (words[sbs_adaptive_state::index(word_e::cut_contract_tag_bits)] != sbs_adaptive_state::cut_contract_tag) {
      return false;
    }
    for (const auto &field : sbs_adaptive_state::fields) {
      const auto index = sbs_adaptive_state::index(field.word);
      if ((field.name.starts_with("reserved_") && words[index] != sbs_adaptive_state::initial_words[index]) ||
          (field.gpu_encoding != sbs_adaptive_state::gpu_encoding_e::uint_bits && !std::isfinite(scalar(field.word)))) {
        return false;
      }
    }
    const float cut_flags = scalar(word_e::cut_flags);
    const float analysis_flags = scalar(word_e::analysis_flags);
    const float history_state = scalar(word_e::model_input_history_state);
    const float cut_pulse = scalar(word_e::hard_cut_pulse);
    return scalar(word_e::scene_age) >= 0.0f &&
           cut_flags >= 0.0f && cut_flags <= static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) &&
           std::trunc(cut_flags) == cut_flags &&
           analysis_flags >= 0.0f && analysis_flags <= static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) &&
           std::trunc(analysis_flags) == analysis_flags &&
           history_state >= 0.0f && history_state <= 4.0f && std::trunc(history_state) == history_state &&
           (cut_pulse == 0.0f || cut_pulse == 1.0f) &&
           words[sbs_adaptive_state::index(word_e::hard_cut_count)] <= sbs_adaptive_state::counter_max &&
           words[sbs_adaptive_state::index(word_e::empty_raw_count)] <= sbs_adaptive_state::counter_max &&
           words[sbs_adaptive_state::index(word_e::collapsed_raw_count)] <= sbs_adaptive_state::counter_max;
  }

  [[nodiscard]] inline diagnostic_sample_t decode_diagnostics(
    const snapshot_words_t &words,
    const expected_t &expected,
    const bool cut_present,
    const bool outcomes_present,
    const std::chrono::steady_clock::time_point sampled_at
  ) noexcept {
    diagnostic_sample_t result {.expected = expected, .sampled_at = sampled_at};
    if (cut_present) {
      sbs_adaptive_state::words_t cut {};
      std::copy_n(words.begin() + cut_state_begin, cut.size(), cut.begin());
      if (valid_cut_state_words(cut)) {
        result.cut_state = cut;
      } else {
        result.failed = true;
      }
    }
    if (outcomes_present) {
      host_sbs_gpu_outcomes::words_t outcomes {};
      std::copy_n(words.begin() + outcome_counts_begin, outcomes.size(), outcomes.begin());
      result.outcome_counts = host_sbs_gpu_outcomes::decode(outcomes);
      result.failed = result.failed || !result.outcome_counts;
    }
    return result;
  }

  [[nodiscard]] inline bool valid_expected(const expected_t &expected) noexcept {
    return expected.estimator_generation != 0u && expected.publication_sequence != 0u &&
           expected.frame_id != 0u && expected.transaction_token != 0u && expected.domain_tag != 0u &&
           expected.width != 0u && expected.height != 0u &&
           std::isfinite(expected.raw_coordinate_scale) && expected.raw_coordinate_scale > 0.0f &&
           cuda_conditional_graph::authenticated_work_flags(expected.expected_work) &&
           (expected.submission_class == submission_class_e::force_infer ||
            (expected.submission_class == submission_class_e::gpu_undecided &&
             expected.observation_timestamp_us != 0u &&
             expected.baseline_frame_id != 0u && expected.baseline_frame_id < expected.frame_id));
  }

  [[nodiscard]] inline bool matches_publication(const receipt_t &receipt, const expected_t &current) noexcept {
    // The sequence changes before any producer-resource mutation, even if a caller repeats a frame
    // ID. A late receipt is useful evidence, but never authenticates newer borrowed SRV aliases.
    return valid_expected(current) && receipt.expected == current;
  }

  [[nodiscard]] inline std::optional<receipt_t> decode(const snapshot_words_t &words, const expected_t &expected) noexcept {
    if (!valid_expected(expected)) {
      return std::nullopt;
    }
    std::array<std::uint32_t, transaction_word_count> transaction {};
    std::copy_n(words.begin() + transaction_begin, transaction.size(), transaction.begin());
    const auto authenticated = trace::authenticate_receipt(
      transaction, expected.transaction_token, expected.expected_work, expected.submission_class
    );
    if (!authenticated.receipt_valid) {
      return std::nullopt;
    }
    const auto subtitle = trace::classify_subtitle_disposition(
      expected.expected_work, expected.host_subtitle_outcome, authenticated, expected.flags
    );
    if (subtitle == subtitle_disposition_e::invalid) {
      return std::nullopt;
    }
    const auto dispatch_matches = [&](std::size_t offset, std::uint32_t x, std::uint32_t y = 1u) {
      return transaction[offset] == x && transaction[offset + 1u] == y &&
             transaction[offset + 2u] == 1u && transaction[offset + 3u] == 0u;
    };
    const auto grid_x = expected.width / 16u + (expected.width % 16u != 0u);
    const auto grid_y = expected.height / 16u + (expected.height % 16u != 0u);
    if (expected.submission_class == submission_class_e::gpu_undecided) {
      const bool infer = authenticated.depth == depth_disposition_e::infer;
      const auto elements = static_cast<std::uint64_t>(expected.width) * expected.height;
      const auto reduce_groups = static_cast<std::uint32_t>(std::min<std::uint64_t>(64u, (elements + 255u) / 256u));
      // These are the existing finalizer's depth verdicts, stronger than the CUDA branch tag.
      if (!dispatch_matches(16u, infer ? reduce_groups : 0u) ||
          !dispatch_matches(20u, infer ? 1u : 0u) ||
          !dispatch_matches(24u, infer ? grid_x : 0u, infer ? grid_y : 1u) ||
          !dispatch_matches(32u, infer ? expected.width : 0u) ||
          !dispatch_matches(36u, infer ? expected.height : 0u) ||
          !dispatch_matches(40u, infer ? 0u : grid_x, infer ? 1u : grid_y)) {
        return std::nullopt;
      }
    }
    const bool current_subtitle = subtitle == subtitle_disposition_e::optional_ocr || subtitle == subtitle_disposition_e::abstention;
    const bool optional = subtitle == subtitle_disposition_e::optional_ocr;
    if (!dispatch_matches(44u, current_subtitle ? grid_x : 0u, current_subtitle ? grid_y : 1u) ||
        !dispatch_matches(48u, optional ? (v2::subtitle_ocr_output_width / 8u + 31u) / 32u : 0u,
          optional ? (v2::subtitle_ocr_output_height + 3u) / 4u : 1u) ||
        !dispatch_matches(52u, optional ? 1u : 0u) ||
        !dispatch_matches(56u, subtitle == subtitle_disposition_e::abstention ? 1u : 0u) ||
        !dispatch_matches(60u, current_subtitle ? 1u : 0u)) {
      return std::nullopt;
    }
    v2::state_words_t state {};
    std::copy_n(words.begin() + parallax_state_begin, state.size(), state.begin());
    if (!v2::parallax_state_words_are_authenticated(state, expected.raw_coordinate_scale)) {
      return std::nullopt;
    }
    receipt_t result {
      .expected = expected,
      .depth = authenticated.depth,
      .geometry_valid = std::bit_cast<float>(state[v2::frame_valid]) == 1.0f,
      .subtitle = subtitle,
    };
    for (std::size_t i = 0u; i < depth_frame_state_word_count; ++i) {
      if (!std::isfinite(std::bit_cast<float>(words[depth_frame_state_begin + i]))) {
        return std::nullopt;
      }
    }
    const auto depth_validity = std::bit_cast<float>(words[depth_frame_state_begin + 3u]);
    const auto normalization_initialized = std::bit_cast<float>(words[depth_frame_state_begin + 2u]);
    if ((depth_validity != 0.0f && depth_validity != 1.0f && depth_validity != 2.0f) ||
        (normalization_initialized != 0.0f && normalization_initialized != 1.0f) ||
        (depth_validity > 0.0f && normalization_initialized != 1.0f) ||
        (normalization_initialized == 1.0f &&
         std::bit_cast<float>(words[depth_frame_state_begin]) > std::bit_cast<float>(words[depth_frame_state_begin + 1u]))) {
      return std::nullopt;
    }
    result.geometry_valid = result.geometry_valid && depth_validity >= 0.5f;
    const auto owner = words.data() + history_owner_begin;
    if (owner[0u] != 0u) {
      const auto frame = trace::join_u64(owner[2u], owner[3u]);
      const auto timestamp = trace::join_u64(owner[8u], owner[9u]);
      if (owner[0u] != history_owner_tag || owner[1u] != history_owner_schema ||
          trace::join_u64(owner[4u], owner[5u]) != expected.domain_tag ||
          owner[6u] != expected.width || owner[7u] != expected.height ||
          frame == 0u || timestamp == 0u || timestamp > expected.observation_timestamp_us ||
          depth_validity < 0.5f ||
          (authenticated.depth == depth_disposition_e::infer ?
             (frame != expected.frame_id || timestamp != expected.observation_timestamp_us) :
             (frame > expected.baseline_frame_id || frame >= expected.frame_id))) {
        return std::nullopt;
      }
      result.depth_owner_valid = true;
      result.depth_owner_frame_id = frame;
      result.depth_owner_timestamp_us = timestamp;
    } else if (authenticated.depth == depth_disposition_e::reuse) {
      // The stronger depth finalizer cannot authorize reuse without a valid retained owner.
      return std::nullopt;
    }
    const auto locator = words.data() + subtitle_locator_begin;
    if (subtitle != subtitle_disposition_e::suppressed) {
      const auto frame = trace::join_u64(locator[22u], locator[23u]);
      const auto generation = trace::join_u64(locator[10u], locator[11u]);
      constexpr std::uint32_t known_locator_flags = 1u | 2u | 4u | 8u | v2::subtitle_locator_provisional_current_flag;
      if (locator[0u] != v2::subtitle_locator_state_schema || locator[1u] != v2::subtitle_locator_state_tag ||
          (locator[2u] & ~known_locator_flags) != 0u ||
          generation != expected.analysis_generation || locator[27u] != expected.width || locator[28u] != expected.height ||
          locator[4u] > v2::subtitle_locator_rectangle_capacity ||
          locator[12u] > v2::subtitle_locator_rectangle_capacity ||
          locator[20u] > v2::subtitle_locator_rectangle_capacity ||
          frame == 0u || (current_subtitle ? frame != expected.frame_id :
            (!result.depth_owner_valid || frame != result.depth_owner_frame_id))) {
        return std::nullopt;
      }
      const auto condition = words.data() + subtitle_condition_begin;
      if (locator[20u] == 0u) {
        if (!std::all_of(condition, condition + subtitle_condition_word_count, [](auto word) { return word == 0u; })) {
          return std::nullopt;
        }
      } else {
        const bool provisional = (locator[2u] & v2::subtitle_locator_provisional_current_flag) != 0u;
        const auto fade = locator[provisional ? v2::subtitle_locator_provisional_fade_word : 24u];
        const auto target = locator[provisional ? v2::subtitle_locator_provisional_target_word : 18u];
        const auto kinds = (locator[v2::subtitle_locator_kind_word] >> v2::subtitle_locator_current_kind_shift) & v2::subtitle_locator_kind_mask;
        // Same compact publication relation as the existing Dump decoder. Geometry validation
        // and provisional selection stay in their existing GPU producers, never in this receipt.
        if (condition[0u] != v2::subtitle_condition_param_schema || condition[1u] != v2::subtitle_condition_param_tag ||
            condition[2u] != locator[20u] || condition[3u] != kinds ||
            condition[4u] != fade || condition[5u] != target ||
            (fade != 1u && fade != 2u) || !std::isfinite(std::bit_cast<float>(target))) {
          return std::nullopt;
        }
      }
      result.subtitle_frame_id = frame;
      result.subtitle_domain_tag = expected.domain_tag;
    }
    return result;
  }

  // Pure metadata half of one bounded staging/query slot. The D3D owner calls retire only after
  // its query and nonblocking Map succeed. Generation changes do not make an in-flight slot free.
  class slot_state_t {
  public:
    [[nodiscard]] bool reserve(const expected_t &expected) noexcept {
      if (pending_ || !valid_expected(expected)) {
        return false;
      }
      pending_ = expected;
      return true;
    }
    [[nodiscard]] bool pending() const noexcept { return pending_.has_value(); }
    [[nodiscard]] const expected_t *expected() const noexcept { return pending_ ? &*pending_ : nullptr; }
    [[nodiscard]] std::optional<receipt_t> retire(const snapshot_words_t &words, std::uint64_t current_estimator_generation) noexcept {
      if (!pending_) {
        return std::nullopt;
      }
      const auto expected = *pending_;
      pending_.reset();
      return expected.estimator_generation == current_estimator_generation ? decode(words, expected) : std::nullopt;
    }
    // Only after destruction/quarantine of the associated GPU resources, never to recycle a busy slot.
    void reset() noexcept { pending_.reset(); }

  private:
    std::optional<expected_t> pending_;
  };

  static_assert(snapshot_byte_count == 672u);
  static_assert(transaction_word_count == trace::transaction_word_count);
  static_assert(subtitle_locator_word_count == v2::subtitle_locator_header_word_count);
  static_assert(parallax_state_word_count == v2::state_words_t {}.size());
}  // namespace models::host_sbs_gpu_completion_receipt
