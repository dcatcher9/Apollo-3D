/**
 * @file src/host_sbs_adaptive_submission.h
 * @brief Shared live/offline Host SBS device-conditional transaction policy.
 */
#pragma once

#include "host_sbs_gpu_completion_receipt.h"

#include <cstdint>

namespace models {

  /** Version of the completed-publication request policy attested by offline replay artifacts. */
  inline constexpr std::uint32_t gpu_adaptive_transaction_policy_schema = 6u;

  /** Host-only prefilter command for the device-owned adaptive reuse transaction. */
  struct gpu_adaptive_reuse_request {
    // CPU-only prefilters may mark an ordinary, same-domain frame as undecided. The detector then
    // publishes a GPU proposal bound to this nonzero transaction token. Conditional launch is
    // separately runtime-gated; setting this field alone can never skip inference.
    bool authorize_gpu_undecided_reuse = false;
    // Actual depth owner from the latest authenticated final-publication receipt.
    std::uint64_t baseline_frame_id = 0u;
    std::uint64_t gpu_reuse_decision_token = 0u;
    // Nonzero monotonic observation time. This remains host/D3D metadata; the fixed CUDA
    // RQST/CBRG transaction deliberately does not carry it.
    std::uint64_t observation_timestamp_us = 0u;
  };

  enum class gpu_adaptive_submission_class_e : std::uint8_t {
    invalid,
    force_infer,
    gpu_undecided,
  };

  /** Form a candidate only from the current completed publication. A missing receipt never
   * authorizes a wait or a guessed owner. Color continuity belongs to the caller; the actual
   * comparison owner comes exclusively from the authenticated GPU receipt. Age and reuse count
   * never expire that owner; explicit subtitle suppression cannot seed an ordinary joint reuse.
   */
  [[nodiscard]] inline gpu_adaptive_reuse_request make_gpu_adaptive_request(
    const std::uint64_t current_frame_id,
    const bool candidate_authorized,
    const host_sbs_gpu_completion_receipt::receipt_t *publication,
    const host_sbs_gpu_completion_receipt::expected_t &current_publication,
    const std::uint64_t observation_timestamp_us
  ) noexcept {
    if (!candidate_authorized || !publication ||
        !host_sbs_gpu_completion_receipt::matches_publication(*publication, current_publication) ||
        !publication->depth_cache_authorized() ||
        (publication->expected.flags & host_sbs_gpu_trace::record_flag_e::subtitle_suppressed) != 0u ||
        current_frame_id <= publication->expected.frame_id ||
        observation_timestamp_us == 0u ||
        observation_timestamp_us < publication->expected.observation_timestamp_us) {
      return {.observation_timestamp_us = observation_timestamp_us};
    }
    return {
      .authorize_gpu_undecided_reuse = true,
      .baseline_frame_id = publication->depth_owner_frame_id,
      .gpu_reuse_decision_token = current_frame_id,
      .observation_timestamp_us = observation_timestamp_us,
    };
  }

  /** Authenticate what was submitted without keeping a second completion watermark. */
  [[nodiscard]] constexpr gpu_adaptive_submission_class_e classify_gpu_adaptive_submission(
    const std::uint64_t current_frame_id,
    const gpu_adaptive_reuse_request &request,
    const bool force_infer_enqueued,
    const bool gpu_undecided_enqueued
  ) noexcept {
    if (force_infer_enqueued == gpu_undecided_enqueued || current_frame_id == 0u ||
        request.observation_timestamp_us == 0u) {
      return gpu_adaptive_submission_class_e::invalid;
    }
    if (force_infer_enqueued) {
      return gpu_adaptive_submission_class_e::force_infer;
    }
    return request.authorize_gpu_undecided_reuse &&
             request.gpu_reuse_decision_token == current_frame_id &&
             request.baseline_frame_id != 0u && request.baseline_frame_id < current_frame_id ?
             gpu_adaptive_submission_class_e::gpu_undecided :
             gpu_adaptive_submission_class_e::invalid;
  }

  enum class depth_optional_work_mode_e : std::uint8_t {
    ordinary,
    suppress_subtitle,  ///< Publish Base and freeze OCR8/SLR13 for native move/size.
  };

  [[nodiscard]] constexpr bool depth_optional_work_allows_gpu_undecided(
    const depth_optional_work_mode_e work
  ) noexcept {
    return work == depth_optional_work_mode_e::ordinary;
  }

  /** Explicit subtitle eligibility is independent of the GPU similarity decision. */
  [[nodiscard]] constexpr depth_optional_work_mode_e select_depth_optional_work_mode(
    const bool observed_interactive_move_size = false,
    const bool snapshot_debug_inputs = false
  ) noexcept {
    return observed_interactive_move_size && !snapshot_debug_inputs ?
             depth_optional_work_mode_e::suppress_subtitle :
             depth_optional_work_mode_e::ordinary;
  }

}  // namespace models
