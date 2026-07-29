/**
 * @file src/sbs_roi_shape_transition.h
 * @brief Pure lifecycle guard for delayed GPU ROI-shape confirmations.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace models {
  enum class sbs_roi_shape_confirmation_result {
    idle,
    waiting,
    confirmed,
    recover_canonical,
  };

  /**
   * Freeze one controller state while its exact GPU shape request crosses the asynchronous
   * readback ring. A completion from any older source frame cannot authorize teardown.
   */
  class sbs_roi_shape_confirmation_guard {
  public:
    static constexpr unsigned max_capture_opportunities = 12u;

    bool begin(
      std::uint64_t source_frame_id,
      std::uint32_t source_width,
      std::uint32_t source_height,
      bool copy_scheduled
    ) {
      if (
        source_frame_id == 0u ||
        source_width == 0u ||
        source_height == 0u
      ) {
        reset();
        return false;
      }
      source_frame_id_ = source_frame_id;
      source_width_ = source_width;
      source_height_ = source_height;
      copy_scheduled_ = copy_scheduled;
      capture_opportunities_ = 0u;
      return true;
    }

    [[nodiscard]] bool awaiting() const {
      return source_frame_id_ != 0u;
    }

    [[nodiscard]] std::uint64_t source_frame_id() const {
      return source_frame_id_;
    }

    [[nodiscard]] std::uint32_t source_width() const {
      return source_width_;
    }

    [[nodiscard]] std::uint32_t source_height() const {
      return source_height_;
    }

    [[nodiscard]] bool copy_scheduled() const {
      return copy_scheduled_;
    }

    /**
     * Observe one capture opportunity.
     *
     * `sample_valid` includes the exact request ABI/source/backend checks performed by the
     * caller. A fresh but older source frame remains stale evidence and only advances the bounded
     * timeout. Scheduling failure is intentionally orthogonal: a valid fresh completion still
     * confirms even when the next submission failed.
     */
    sbs_roi_shape_confirmation_result observe(
      bool scheduled_now,
      bool fresh_sample,
      std::uint64_t sampled_source_frame_id,
      bool sample_valid
    ) {
      if (!awaiting()) {
        return sbs_roi_shape_confirmation_result::idle;
      }
      copy_scheduled_ = copy_scheduled_ || scheduled_now;
      if (
        fresh_sample &&
        sampled_source_frame_id == source_frame_id_ &&
        sample_valid
      ) {
        reset();
        return sbs_roi_shape_confirmation_result::confirmed;
      }
      if (++capture_opportunities_ >= max_capture_opportunities) {
        reset();
        return sbs_roi_shape_confirmation_result::recover_canonical;
      }
      return sbs_roi_shape_confirmation_result::waiting;
    }

    void reset() {
      source_frame_id_ = 0u;
      source_width_ = 0u;
      source_height_ = 0u;
      copy_scheduled_ = false;
      capture_opportunities_ = 0u;
    }

  private:
    std::uint64_t source_frame_id_ = 0u;
    std::uint32_t source_width_ = 0u;
    std::uint32_t source_height_ = 0u;
    bool copy_scheduled_ = false;
    unsigned capture_opportunities_ = 0u;
  };

  enum class sbs_roi_shape_binding_failure_action {
    recover_canonical,
    retire_estimator,
  };

  enum class sbs_trt_context_disposition {
    reuse,
    quarantine,
  };

  enum class sbs_cuda_resource_cleanup_disposition {
    release,
    retain_until_process_exit,
  };

  /**
   * Context-bound CUDA handles may be released only after selecting their owning context and
   * proving the stream idle. On either failure the bounded terminal path intentionally leaks the
   * handles until process exit; attempting cleanup under uncertainty can free memory still used
   * by TensorRT or call into the wrong CUDA context.
   */
  [[nodiscard]] constexpr sbs_cuda_resource_cleanup_disposition
  sbs_cuda_resource_cleanup_policy(
    bool owning_context_selected,
    bool stream_idle
  ) {
    return owning_context_selected && stream_idle ?
             sbs_cuda_resource_cleanup_disposition::release :
             sbs_cuda_resource_cleanup_disposition::
               retain_until_process_exit;
  }

  enum class sbs_trt_context_event {
    none,
    recoverable_dynamic_shape_rejection,
    canonical_shape_rejection,
    tensor_address_rejection,
    enqueue_rejection,
    cuda_context_failure,
    cuda_stream_failure,
    cuda_interop_failure,
  };

  /**
   * Runtime context health is monotonic. A rejected noncanonical shape is recoverable only because
   * the owner rebuilds and explicitly rebinds the canonical shape; every context/device/stream or
   * interop failure permanently removes the execution context from the reusable pool.
   */
  [[nodiscard]] constexpr bool sbs_trt_context_reusable_after(
    bool currently_reusable,
    sbs_trt_context_event event
  ) {
    if (!currently_reusable) {
      return false;
    }
    return
      event == sbs_trt_context_event::none ||
      event ==
        sbs_trt_context_event::recoverable_dynamic_shape_rejection;
  }

  struct sbs_trt_context_pool_accounting {
    std::size_t usable = 0u;
    std::size_t warmed = 0u;
    std::size_t quarantined = 0u;
  };

  /**
   * Quarantine moves one physically allocated context out of usable accounting without changing
   * the total allocation. Saturation keeps this helper fail-safe if a terminal path is observed
   * after partially constructed accounting.
   */
  [[nodiscard]] constexpr sbs_trt_context_pool_accounting
  sbs_trt_context_accounting_after_quarantine(
    sbs_trt_context_pool_accounting value,
    bool was_warmed
  ) {
    if (value.usable > 0u) {
      --value.usable;
    }
    if (was_warmed && value.warmed > 0u) {
      --value.warmed;
    }
    ++value.quarantined;
    return value;
  }

  /**
   * A warmed TensorRT execution context is reusable only while every shape/binding/enqueue and
   * terminal CUDA operation observed by its owner remained healthy. TensorRT interfaces cannot
   * be destroyed safely across the MinGW/MSVC boundary, so failed contexts are leaked in a
   * bounded quarantine rather than returned to an unsuspecting later stream.
   */
  [[nodiscard]] constexpr sbs_trt_context_disposition
  sbs_trt_context_pool_disposition(
    bool context_warmed,
    bool context_reusable
  ) {
    return context_warmed && context_reusable ?
             sbs_trt_context_disposition::reuse :
             sbs_trt_context_disposition::quarantine;
  }

  enum class sbs_roi_shape_sample_action {
    continue_current_shape,
    confirm_shape_change,
    apply_exact_shape_change,
  };

  /**
   * A delayed request that keeps the current tensor dimensions is non-destructive: the GPU
   * transform builder still validates its ROI generation and committed rectangle before it can
   * activate the crop, so inference can continue without a CPU-readback bubble. A request that
   * changes tensor dimensions must be confirmed against the frozen current controller frame
   * before resources are torn down.
   */
  [[nodiscard]] constexpr sbs_roi_shape_sample_action
  sbs_roi_shape_sample_transition(
    std::uint64_t sampled_source_frame_id,
    std::uint64_t current_source_frame_id,
    std::uint32_t sampled_model_width,
    std::uint32_t sampled_model_height,
    std::uint32_t current_model_width,
    std::uint32_t current_model_height
  ) {
    if (
      sampled_model_width == current_model_width &&
      sampled_model_height == current_model_height
    ) {
      return sbs_roi_shape_sample_action::continue_current_shape;
    }
    return sampled_source_frame_id == current_source_frame_id ?
             sbs_roi_shape_sample_action::apply_exact_shape_change :
             sbs_roi_shape_sample_action::confirm_shape_change;
  }

  /**
   * Withdrawing active crop authority always rebuilds the shape-dependent histories. This is
   * intentionally independent of tensor dimensions: two crops can share an identical model shape
   * while their depth, normalization, and subject histories use different coordinates.
   */
  [[nodiscard]] constexpr bool
  sbs_roi_canonical_recovery_requires_rebuild(
    bool active_roi_authority,
    bool recovery_requested
  ) {
    return active_roi_authority && recovery_requested;
  }

  [[nodiscard]] constexpr sbs_roi_shape_binding_failure_action
  sbs_roi_shape_binding_failure(
    bool active_roi_authority,
    std::uint32_t current_width,
    std::uint32_t current_height,
    std::uint32_t canonical_width,
    std::uint32_t canonical_height
  ) {
    return
      active_roi_authority &&
      (current_width != canonical_width ||
       current_height != canonical_height) ?
        sbs_roi_shape_binding_failure_action::recover_canonical :
        sbs_roi_shape_binding_failure_action::retire_estimator;
  }
}  // namespace models
