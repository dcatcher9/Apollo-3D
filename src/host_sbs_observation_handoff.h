/**
 * @file src/host_sbs_observation_handoff.h
 * @brief Compose observation completion and current submission at the converter boundary.
 */
#pragma once

#include "video_depth_estimator.h"

#include <optional>
#include <utility>

namespace host_sbs {
  class observation_handoff_t {
  public:
    observation_handoff_t() = default;

    // Copy the views with their verdict. Moving the views out would leave a verdict on the
    // moved-from selection, whose view tuple would then be incomplete.
    observation_handoff_t(const observation_handoff_t &) = default;
    observation_handoff_t &operator=(const observation_handoff_t &source) {
      if (this != &source) {
        select(source.selected_);
        structural_authentication_ = source.structural_authentication_;
      }
      return *this;
    }

    const models::depth_step_result_t &step() const noexcept {
      return selected_;
    }

    void select(models::depth_step_result_t result) {
      structural_authentication_.reset();
      selected_ = std::move(result);
    }

    void reset() {
      select({});
    }

    // This validates immutable resource descriptors and copied observation metadata.
    // Publication currentness, route authority and GPU readiness belong to the caller.
    [[nodiscard]] bool authenticate_structure() const {
      if (!selected_.observation.completed_frame_valid) {
        return false;
      }
      if (!structural_authentication_) {
        structural_authentication_ = models::parallax_v2_result_is_authenticated(selected_.observation);
      }
      return has_authenticated_structure();
    }

    [[nodiscard]] bool has_authenticated_structure() const noexcept {
      return structural_authentication_.value_or(false);
    }

    /** The caller authenticates and finalizes the selected completion, and revokes that verdict
     * if later admission invalidates its publication. This composition grants no GPU authority.
     * Return whether the already-finalized completion was preserved for this delivery.
     */
    bool accept_submission(models::depth_step_result_t submitted, const bool selected_completion_finalized) {
      if (selected_completion_finalized && selected_.observation.completed_frame_valid && !submitted.observation.completed_frame_valid) {
        selected_.submission = submitted.submission;
        selected_.status.cuda_graph_active =
          selected_.status.cuda_graph_active || submitted.status.cuda_graph_active;
        return true;
      }
      select(std::move(submitted));
      return false;
    }

    /** Retained views and diagnostic status do not replay submission or completion events. */
    void select_retained(const observation_handoff_t &source) {
      const auto authentication = source.structural_authentication_;
      structural_authentication_.reset();
      selected_.observation = source.selected_.observation;
      selected_.status = source.selected_.status;
      selected_.submission = {};
      selected_.completion = {};
      structural_authentication_ = authentication;
    }

  private:
    models::depth_step_result_t selected_;
    mutable std::optional<bool> structural_authentication_;
  };
}  // namespace host_sbs
