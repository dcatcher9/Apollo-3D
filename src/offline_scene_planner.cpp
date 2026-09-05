/**
 * @file src/offline_scene_planner.cpp
 * @brief Causal scene epochs derived from the authenticated online cut state.
 */
#include "offline_scene_planner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace offline_sbs {
  namespace {
    constexpr float depth_cut_high = 0.60f;
    constexpr float depth_cut_corroborate = 0.25f;

    bool finite(float value) {
      return std::isfinite(value);
    }

    bool finite(double value) {
      return std::isfinite(value);
    }

    bool usable_metric(const std::optional<float> &value) {
      return value && finite(*value) && *value >= 0.0f;
    }

    float metric_or_zero(const std::optional<float> &value) {
      return usable_metric(value) ? *value : 0.0f;
    }

    bool has_flag(const scene_frame_t &frame, std::uint32_t flag) {
      return (frame.analysis_flags & flag) != 0;
    }

    bool appearance_qualified(const scene_frame_t &frame) {
      if (has_flag(frame, analysis_appearance_veto)) {
        return false;
      }
      return usable_metric(frame.depth_change_fraction) &&
             *frame.depth_change_fraction >= depth_cut_corroborate &&
             has_flag(frame, analysis_appearance_proposal);
    }

    bool structureless_geometry_exception(const scene_frame_t &frame) {
      return
        has_flag(frame, analysis_structureless_transition) &&
        has_flag(frame, analysis_geometry_confirmation_candidate);
    }

    bool geometry_structure_corroborated(const scene_frame_t &frame) {
      return
        structureless_geometry_exception(frame) ||
        (
          usable_metric(frame.structural_change_fraction) &&
          *frame.structural_change_fraction >=
            structural_geometry_cut_floor
        );
    }

    bool geometry_qualified(const scene_frame_t &frame) {
      return
        !has_flag(frame, analysis_appearance_veto) &&
        geometry_structure_corroborated(frame) &&
        usable_metric(frame.depth_change_fraction) &&
        *frame.depth_change_fraction >= depth_cut_high;
    }

    bool relative_geometry_qualified(const scene_frame_t &frame) {
      return
        !has_flag(frame, analysis_appearance_veto) &&
        geometry_structure_corroborated(frame) &&
        has_flag(frame, analysis_relative_geometry_spike);
    }

    float evidence_score(const scene_frame_t &frame) {
      const auto depth = metric_or_zero(frame.depth_change_fraction);
      const auto geometry = depth / depth_cut_high;
      const auto appearance =
        has_flag(frame, analysis_appearance_proposal) ?
          depth / depth_cut_corroborate :
          0.0f;
      return std::max(geometry, appearance);
    }
  }  // namespace

  void causal_scene_tracker_t::append(scene_frame_t frame) {
    if (!first_frame_) {
      first_frame_ = frame;
    }
    last_frame_ = frame;
    ++frame_count_;
    if (frame.depth_updated) {
      ++depth_update_count_;
    }
    if (has_flag(frame, analysis_appearance_veto)) {
      ++appearance_veto_count_;
    }
    if (usable_metric(frame.depth_change_fraction) &&
        (!depth_change_max_ ||
         *frame.depth_change_fraction > *depth_change_max_)) {
      depth_change_max_ = frame.depth_change_fraction;
    }
  }

  scene_plan_t causal_scene_tracker_t::finalize_open(
    boundary_audit_t boundary
  ) {
    if (!first_frame_ || !last_frame_ || frame_count_ == 0) {
      throw scene_plan_error("cannot finalize an empty causal scene");
    }

    scene_plan_t result;
    result.scene_id = ++scene_number_;
    result.semantic_scene_id = semantic_scene_number_;
    result.start_sequence = first_frame_->sequence;
    result.end_sequence_exclusive = last_frame_->sequence + 1;
    result.frame_count = frame_count_;
    result.evidence.source_frame_count = frame_count_;
    result.evidence.depth_update_count = depth_update_count_;
    result.evidence.appearance_veto_count = appearance_veto_count_;
    result.evidence.depth_change_max = depth_change_max_;
    result.boundary = std::move(boundary);
    result.cut_state_semantics = "causal-production-exact";
    result.known_limit =
      "diagnostic scene epochs only; rendering is committed causally per source frame";

    if (first_frame_->pts_seconds && last_frame_->pts_seconds &&
        last_frame_->duration_seconds &&
        finite(*first_frame_->pts_seconds) &&
        finite(*last_frame_->pts_seconds) &&
        finite(*last_frame_->duration_seconds) &&
        *last_frame_->duration_seconds > 0.0 &&
        *last_frame_->pts_seconds >= *first_frame_->pts_seconds) {
      const auto end =
        *last_frame_->pts_seconds + *last_frame_->duration_seconds;
      if (finite(end)) {
        result.start_pts_seconds = *first_frame_->pts_seconds;
        result.end_pts_seconds_exclusive = end;
      }
    }

    if (result.boundary.semantic_cut) {
      ++semantic_scene_number_;
    }
    first_frame_.reset();
    last_frame_.reset();
    frame_count_ = 0;
    depth_update_count_ = 0;
    appearance_veto_count_ = 0;
    depth_change_max_.reset();
    return result;
  }

  std::vector<scene_plan_t> causal_scene_tracker_t::feed(
    scene_frame_t frame
  ) {
    if (closed_) {
      throw scene_plan_error("cannot feed a finalized causal scene tracker");
    }
    if (frame.sequence != next_sequence_) {
      throw scene_plan_error(
        "causal scene tracker requires contiguous one-based sequences"
      );
    }
    if (frame.sequence == std::numeric_limits<std::uint64_t>::max()) {
      throw scene_plan_error("causal scene source sequence overflows");
    }
    if (frame.frame_id.empty() ||
        frame.frame_id.size() > max_scene_frame_id_bytes) {
      throw scene_plan_error("causal scene frame identifier is invalid");
    }

    const auto validate_fraction = [](
      const std::optional<float> &value,
      const char *name
    ) {
      if (value && (!finite(*value) || *value < 0.0f || *value > 1.0f)) {
        throw scene_plan_error(std::string {name} + " must be finite in [0, 1]");
      }
    };
    validate_fraction(frame.depth_change_fraction, "depth_change_fraction");
    validate_fraction(frame.raw_rgb_change_fraction, "raw_rgb_change_fraction");
    validate_fraction(
      frame.structural_change_fraction,
      "structural_change_fraction"
    );
    validate_fraction(
      frame.current_structural_support_fraction,
      "current_structural_support_fraction"
    );
    validate_fraction(
      frame.previous_structural_support_fraction,
      "previous_structural_support_fraction"
    );
    validate_fraction(
      frame.common_structural_support_fraction,
      "common_structural_support_fraction"
    );
    if (frame.pts_seconds && !finite(*frame.pts_seconds)) {
      throw scene_plan_error("causal scene PTS must be finite");
    }
    if (frame.duration_seconds &&
        (!finite(*frame.duration_seconds) || *frame.duration_seconds <= 0.0)) {
      throw scene_plan_error("causal scene duration must be positive and finite");
    }

    if (frame.sequence == 1) {
      if (frame.hard_cut_pulse || frame.hard_cut_count != 0) {
        throw scene_plan_error("causal cut state must begin at generation zero");
      }
    } else {
      const auto expected_count =
        frame.hard_cut_pulse &&
        previous_hard_cut_count_ < sbs_adaptive_state::counter_max ?
          previous_hard_cut_count_ + 1u :
          previous_hard_cut_count_;
      if (frame.hard_cut_count != expected_count) {
        throw scene_plan_error(
          "causal hard-cut pulse/count transition is inconsistent"
        );
      }
    }

    std::vector<scene_plan_t> finalized;
    if (frame.hard_cut_pulse) {
      if (!first_frame_) {
        throw scene_plan_error("causal hard cut cannot precede the first frame");
      }
      boundary_audit_t boundary;
      boundary.proposal_sequences.push_back(frame.sequence);
      boundary.proposal_frame_ids.push_back(frame.frame_id);
      boundary.final_sequence = frame.sequence;
      boundary.decision = boundary_decision_e::confirmed;
      boundary.reason =
        "authenticated online hard-cut pulse/count starts the next causal scene";
      boundary.accepted = true;
      boundary.semantic_cut = true;
      boundary.candidate_count = 1;
      boundary.selected_sequence = frame.sequence;
      boundary.selected_frame_id = frame.frame_id;
      boundary.selected_depth_change_fraction = frame.depth_change_fraction;
      boundary.selected_raw_rgb_change_fraction =
        frame.raw_rgb_change_fraction;
      boundary.selected_structural_change_fraction =
        frame.structural_change_fraction;
      boundary.selected_appearance_qualified = appearance_qualified(frame);
      boundary.selected_geometry_qualified = geometry_qualified(frame);
      boundary.selected_relative_geometry_spike =
        relative_geometry_qualified(frame);
      boundary.evidence_window_first_sequence = frame.sequence;
      boundary.evidence_window_last_sequence = frame.sequence;
      if (usable_metric(frame.depth_change_fraction)) {
        boundary.selected_evidence_score = evidence_score(frame);
      }
      boundary_audit_.push_back(boundary);
      finalized.push_back(finalize_open(std::move(boundary)));
    }

    append(std::move(frame));
    previous_hard_cut_count_ = last_frame_->hard_cut_count;
    ++next_sequence_;
    return finalized;
  }

  std::vector<scene_plan_t> causal_scene_tracker_t::finish() {
    if (closed_) {
      throw scene_plan_error("causal scene tracker was already finalized");
    }
    if (!first_frame_) {
      throw scene_plan_error("cannot finalize an empty causal clip");
    }

    boundary_audit_t boundary;
    boundary.accepted = true;
    boundary.semantic_cut = false;
    boundary.decision = boundary_decision_e::end_of_stream;
    boundary.reason = "the final causal scene ends at the source EOF";
    closed_ = true;
    return {finalize_open(std::move(boundary))};
  }

  const std::vector<boundary_audit_t> &
  causal_scene_tracker_t::boundary_audit() const {
    return boundary_audit_;
  }

  std::vector<boundary_audit_t> causal_scene_tracker_t::take_boundary_audit() {
    std::vector<boundary_audit_t> result;
    result.swap(boundary_audit_);
    return result;
  }

  const char *boundary_decision_name(boundary_decision_e decision) {
    switch (decision) {
      case boundary_decision_e::confirmed:
        return "confirmed";
      case boundary_decision_e::moved_to_correlated_evidence:
        return "moved_to_correlated_evidence";
      case boundary_decision_e::merged_duplicate_proposals:
        return "merged_duplicate_proposals";
      case boundary_decision_e::confirmed_causal_fallback:
        return "confirmed_causal_fallback";
      case boundary_decision_e::rejected_supported_flash_return:
        return "rejected_supported_flash_return";
      case boundary_decision_e::rejected_unsupported_proposal:
        return "rejected_unsupported_proposal";
      case boundary_decision_e::rejected_minimum_scene_length:
        return "rejected_minimum_scene_length";
      case boundary_decision_e::administrative_cache_split:
        return "administrative_cache_split";
      case boundary_decision_e::end_of_stream:
        return "end_of_stream";
    }
    return "unknown";
  }
}  // namespace offline_sbs
