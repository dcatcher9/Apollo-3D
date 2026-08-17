/**
 * @file src/host_sbs_adaptive_submission.h
 * @brief Shared live/offline Host SBS device-conditional transaction policy.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace models {

  /** Version of the request/chaining policy attested by offline replay artifacts. */
  inline constexpr std::uint32_t gpu_adaptive_transaction_policy_schema = 2u;
  inline constexpr std::uint32_t gpu_adaptive_max_infer_owner_frame_age = 4u;
  inline constexpr std::uint64_t gpu_adaptive_max_infer_owner_observation_age_us = 100000u;
  inline constexpr std::uint64_t gpu_adaptive_ocr_max_observation_age_us = 33000u;
  inline constexpr std::uint32_t gpu_adaptive_ocr_max_dirty_holds = 2u;

  /** Host-only prefilter command for the device-owned adaptive reuse transaction. */
  struct gpu_adaptive_reuse_request {
    // CPU-only prefilters may mark an ordinary, same-domain frame as undecided. The detector then
    // publishes a GPU proposal bound to this nonzero transaction token. Conditional launch is
    // separately runtime-gated; setting this field alone can never skip inference.
    bool authorize_gpu_undecided_reuse = false;
    // A follow-up may name the immediately preceding opaque transaction instead of the last
    // CPU-known infer. The device-owned history owner still decides whether comparison is legal:
    // infer advances that owner, while reuse retains it so a bounded-age follow-up may compare
    // cumulatively. Invalid or over-age ownership forces infer without exposing the prior branch
    // to the CPU.
    bool opaque_followup = false;
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

  enum class depth_optional_work_mode_e : std::uint8_t {
    ordinary,
    ordinary_due,  ///< Run an independent exact OCR observation on infer or depth reuse.
    suppress_subtitle,  ///< Publish Base and freeze OCR8/SLR13 for native move/size.
  };

  [[nodiscard]] constexpr bool depth_optional_work_allows_gpu_undecided(
    const depth_optional_work_mode_e work
  ) noexcept {
    return work == depth_optional_work_mode_e::ordinary ||
           work == depth_optional_work_mode_e::ordinary_due;
  }

  /** Common request, chaining, and completion policy for live and offline execution.
   *
   * Capture-specific admission stays outside this class: production proves DDup damage, route,
   * authority, and freshness, while an offline replay supplies an already-decoded ordered corpus.
   * Once either caller admits a candidate, this is the single owner of request formation and the
   * opaque observation watermark. Neither caller may infer the private device branch.
   */
  class gpu_adaptive_transaction_policy_t {
  public:
    constexpr void reset() noexcept {
      conditional_frame_id_ = 0u;
    }

    /** Form an estimator request from caller-owned admission evidence.
     *
     * An inactive policy accepts only an initial request against a CPU-known infer baseline. An
     * active policy accepts only an opaque request against the immediately preceding conditional
     * submission. Any malformed or stale authority fails open to the estimator's force path.
     */
    [[nodiscard]] constexpr gpu_adaptive_reuse_request make_request(
      const std::uint64_t current_frame_id,
      const bool candidate_authorized,
      const bool candidate_is_opaque_followup,
      const std::uint64_t candidate_baseline_frame_id,
      const std::uint64_t observation_timestamp_us
    ) const noexcept {
      if (
        !candidate_authorized || current_frame_id == 0u || observation_timestamp_us == 0u ||
        candidate_baseline_frame_id == 0u ||
        current_frame_id <= candidate_baseline_frame_id ||
        candidate_is_opaque_followup != active() ||
        (candidate_is_opaque_followup &&
         candidate_baseline_frame_id != conditional_frame_id_)
      ) {
        // A CPU-known force still owns the current observation time. The GPU infer-gated history
        // owner needs it to seed the next cumulative candidate even though no adaptive request is
        // authorized for this frame.
        return {.observation_timestamp_us = observation_timestamp_us};
      }
      return {
        .authorize_gpu_undecided_reuse = true,
        .opaque_followup = candidate_is_opaque_followup,
        .baseline_frame_id = candidate_baseline_frame_id,
        .gpu_reuse_decision_token = current_frame_id,
        .observation_timestamp_us = observation_timestamp_us,
      };
    }

    /** Authenticate the CPU-visible submission class and advance only its opaque watermark. */
    [[nodiscard]] constexpr gpu_adaptive_submission_class_e record_submission(
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
      if (
        !request.authorize_gpu_undecided_reuse ||
        request.gpu_reuse_decision_token != current_frame_id ||
        request.observation_timestamp_us == 0u ||
        request.baseline_frame_id == 0u ||
        request.baseline_frame_id >= current_frame_id ||
        request.opaque_followup != active() ||
        (request.opaque_followup &&
         request.baseline_frame_id != conditional_frame_id_)
      ) {
        return gpu_adaptive_submission_class_e::invalid;
      }
      record_gpu_undecided_enqueue(current_frame_id);
      return gpu_adaptive_submission_class_e::gpu_undecided;
    }

    /** Restore or advance the watermark when a caller has separately authenticated ownership. */
    constexpr void record_gpu_undecided_enqueue(
      const std::uint64_t frame_id
    ) noexcept {
      if (frame_id != 0u) {
        conditional_frame_id_ = std::max(conditional_frame_id_, frame_id);
      }
    }

    /** Release opaque attribution only for a newer accepted CPU-known force completion. */
    [[nodiscard]] constexpr bool record_known_force_infer_completion(
      const std::uint64_t frame_id,
      const bool accepted
    ) noexcept {
      if (
        conditional_frame_id_ == 0u || !accepted ||
        frame_id <= conditional_frame_id_
      ) {
        return false;
      }
      reset();
      return true;
    }

    [[nodiscard]] constexpr bool active() const noexcept {
      return conditional_frame_id_ != 0u;
    }

    [[nodiscard]] constexpr std::uint64_t conditional_frame_id() const noexcept {
      return conditional_frame_id_;
    }

  private:
    std::uint64_t conditional_frame_id_ = 0u;
  };

  /** Host-known upper bound for independent subtitle observations.
   *
   * Ordinary opaque roots may reuse depth and therefore cannot be credited as OCR observations.
   * At most two such dirty roots, or 33 ms of source observation time, may be accepted before a
   * branch-independent due transaction.
   */
  class gpu_adaptive_ocr_cadence_t {
  public:
    constexpr void reset() noexcept {
      last_guaranteed_observation_us_ = 0u;
      accepted_dirty_holds_ = 0u;
    }

    [[nodiscard]] constexpr bool due(
      const std::uint64_t observation_timestamp_us
    ) const noexcept {
      return observation_timestamp_us == 0u || last_guaranteed_observation_us_ == 0u ||
             observation_timestamp_us < last_guaranteed_observation_us_ ||
             observation_timestamp_us - last_guaranteed_observation_us_ >=
               gpu_adaptive_ocr_max_observation_age_us ||
             accepted_dirty_holds_ >= gpu_adaptive_ocr_max_dirty_holds;
    }

    [[nodiscard]] constexpr depth_optional_work_mode_e select_mode(
      const std::uint64_t observation_timestamp_us,
      const bool observed_interactive_move_size = false,
      const bool snapshot_debug_inputs = false
    ) const noexcept {
      if (snapshot_debug_inputs) {
        return depth_optional_work_mode_e::ordinary;
      }
      if (observed_interactive_move_size) {
        return depth_optional_work_mode_e::suppress_subtitle;
      }
      return due(observation_timestamp_us) ? depth_optional_work_mode_e::ordinary_due :
                                             depth_optional_work_mode_e::ordinary;
    }

    constexpr void record_guaranteed(
      const std::uint64_t observation_timestamp_us
    ) noexcept {
      if (observation_timestamp_us == 0u ||
          (last_guaranteed_observation_us_ != 0u &&
           observation_timestamp_us < last_guaranteed_observation_us_)) {
        reset();
        return;
      }
      last_guaranteed_observation_us_ = observation_timestamp_us;
      accepted_dirty_holds_ = 0u;
    }

    constexpr void record_dirty_hold() noexcept {
      accepted_dirty_holds_ = std::min(
        accepted_dirty_holds_ + 1u,
        gpu_adaptive_ocr_max_dirty_holds
      );
    }

    constexpr void record_accepted(
      const depth_optional_work_mode_e mode,
      const gpu_adaptive_submission_class_e submission_class,
      const std::uint64_t observation_timestamp_us
    ) noexcept {
      if (submission_class == gpu_adaptive_submission_class_e::invalid) {
        return;
      }
      if (mode == depth_optional_work_mode_e::suppress_subtitle) {
        reset();
      } else if (mode == depth_optional_work_mode_e::ordinary_due ||
                 submission_class == gpu_adaptive_submission_class_e::force_infer) {
        record_guaranteed(observation_timestamp_us);
      } else {
        record_dirty_hold();
      }
    }

    [[nodiscard]] constexpr std::uint64_t last_guaranteed_observation_us() const noexcept {
      return last_guaranteed_observation_us_;
    }

    [[nodiscard]] constexpr std::uint32_t accepted_dirty_holds() const noexcept {
      return accepted_dirty_holds_;
    }

  private:
    std::uint64_t last_guaranteed_observation_us_ = 0u;
    std::uint32_t accepted_dirty_holds_ = 0u;
  };

}  // namespace models
