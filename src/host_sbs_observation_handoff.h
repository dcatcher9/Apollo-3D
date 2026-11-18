/**
 * @file src/host_sbs_observation_handoff.h
 * @brief Compose observation completion and current submission at the converter boundary.
 */
#pragma once

#include "video_depth_estimator.h"

#include <utility>

namespace host_sbs {
  class observation_handoff_t {
  public:
    models::depth_step_result_t &step() noexcept {
      return selected_;
    }

    const models::depth_step_result_t &step() const noexcept {
      return selected_;
    }

    void select(models::depth_step_result_t result) {
      selected_ = std::move(result);
    }

    void reset() {
      selected_ = {};
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
      selected_ = std::move(submitted);
      return false;
    }

    /** Retained views and diagnostic status do not replay submission or completion events. */
    static models::depth_step_result_t repeat(
      const models::depth_observation_views_t &observation,
      const models::depth_call_status_t &status
    ) {
      return {
        .observation = observation,
        .submission = {},
        .completion = {},
        .status = status,
      };
    }

  private:
    models::depth_step_result_t selected_;
  };
}  // namespace host_sbs
