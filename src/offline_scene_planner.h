/**
 * @file src/offline_scene_planner.h
 * @brief Causal scene epochs derived from the authenticated online cut state.
 */
#pragma once

#include "generated/sbs_adaptive_state_contract.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace offline_sbs {
  inline constexpr std::size_t max_scene_frame_id_bytes = 64;
  // Keep this named C++ policy in lockstep with the production shader's
  // STRUCTURAL_GEOMETRY_CUT_FLOOR. HLSL and C++ cannot consume one common
  // declaration without introducing a generated cross-language contract.
  inline constexpr float structural_geometry_cut_floor = 0.005f;

  inline constexpr std::uint32_t analysis_appearance_proposal =
    sbs_adaptive_state::analysis_flag_appearance_proposal;
  inline constexpr std::uint32_t analysis_exposure_like =
    sbs_adaptive_state::analysis_flag_exposure_like;
  inline constexpr std::uint32_t analysis_structureless_transition =
    sbs_adaptive_state::analysis_flag_structureless;
  inline constexpr std::uint32_t analysis_same_scene_return =
    sbs_adaptive_state::analysis_flag_same_return;
  inline constexpr std::uint32_t analysis_appearance_veto =
    sbs_adaptive_state::analysis_flag_veto;
  inline constexpr std::uint32_t analysis_relative_geometry_spike =
    sbs_adaptive_state::analysis_flag_relative_spike;
  inline constexpr std::uint32_t analysis_geometry_confirmation_candidate =
    sbs_adaptive_state::analysis_flag_geometry_confirmation_candidate;

  /**
   * One source-frame sample from the native adaptive trace.
   *
   * `sequence` is one-based and contiguous. The authenticated hard-cut pulse
   * and count are authoritative; the exported metrics are retained only for
   * audit summaries of the causal epoch that the online state already chose.
   */
  struct scene_frame_t {
    std::uint64_t sequence = 0;
    std::string frame_id;
    bool depth_updated = false;
    bool hard_cut_pulse = false;
    std::uint32_t hard_cut_count = 0;
    std::optional<float> depth_change_fraction;
    std::optional<float> raw_rgb_change_fraction;
    std::optional<float> structural_change_fraction;
    std::optional<float> current_structural_support_fraction;
    std::optional<float> previous_structural_support_fraction;
    std::optional<float> common_structural_support_fraction;
    std::uint32_t analysis_flags = 0;
    std::optional<double> pts_seconds;
    std::optional<double> duration_seconds;
  };

  enum class boundary_decision_e {
    confirmed,
    moved_to_correlated_evidence,
    merged_duplicate_proposals,
    confirmed_causal_fallback,
    rejected_supported_flash_return,
    rejected_unsupported_proposal,
    rejected_minimum_scene_length,
    administrative_cache_split,
    end_of_stream,
  };

  struct boundary_audit_t {
    std::vector<std::uint64_t> proposal_sequences;
    std::vector<std::string> proposal_frame_ids;
    std::optional<std::uint64_t> final_sequence;
    boundary_decision_e decision = boundary_decision_e::confirmed;
    std::string reason;
    bool accepted = false;
    bool semantic_cut = false;
    bool truncated = false;
    bool budget_forced = false;
    std::int64_t revision_depth_updates = 0;
    std::int64_t revision_source_frames = 0;
    std::size_t candidate_count = 0;
    std::optional<float> selected_evidence_score;
    std::optional<std::uint64_t> evidence_window_first_sequence;
    std::optional<std::uint64_t> evidence_window_last_sequence;
    std::optional<std::uint64_t> selected_sequence;
    std::optional<std::string> selected_frame_id;
    std::optional<float> selected_depth_change_fraction;
    std::optional<float> selected_raw_rgb_change_fraction;
    std::optional<float> selected_structural_change_fraction;
    bool selected_appearance_qualified = false;
    bool selected_geometry_qualified = false;
    bool selected_relative_geometry_spike = false;
  };

  struct scene_evidence_t {
    std::size_t source_frame_count = 0;
    std::size_t depth_update_count = 0;
    std::size_t appearance_veto_count = 0;
    std::optional<float> depth_change_max;
  };

  struct scene_plan_t {
    std::uint64_t scene_id = 0;
    std::uint64_t semantic_scene_id = 0;
    std::uint64_t start_sequence = 0;
    std::uint64_t end_sequence_exclusive = 0;
    std::uint64_t frame_count = 0;
    std::uint64_t cache_bytes = 0;
    std::optional<double> start_pts_seconds;
    std::optional<double> end_pts_seconds_exclusive;
    scene_evidence_t evidence;
    boundary_audit_t boundary;
    bool ground_truth = false;
    std::string cut_state_semantics = "causal-production-exact";
    std::string known_limit =
      "diagnostic scene epochs only; rendering is committed causally per source frame";
  };

  class scene_plan_error: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  /**
   * Builds diagnostic scene epochs from the authenticated production cut state
   * without revising it. A cut frame starts the new epoch, exactly as the
   * online camera state does. Only aggregate evidence for the open epoch is
   * retained.
   */
  class causal_scene_tracker_t {
  public:
    causal_scene_tracker_t() = default;

    std::vector<scene_plan_t> feed(scene_frame_t frame);
    std::vector<scene_plan_t> finish();

    [[nodiscard]] const std::vector<boundary_audit_t> &boundary_audit() const;

  private:
    std::vector<boundary_audit_t> boundary_audit_;
    std::optional<scene_frame_t> first_frame_;
    std::optional<scene_frame_t> last_frame_;
    std::uint64_t next_sequence_ = 1;
    std::uint64_t scene_number_ = 0;
    std::uint64_t semantic_scene_number_ = 1;
    std::size_t frame_count_ = 0;
    std::size_t depth_update_count_ = 0;
    std::size_t appearance_veto_count_ = 0;
    std::optional<float> depth_change_max_;
    std::uint32_t previous_hard_cut_count_ = 0;
    bool closed_ = false;

    void append(scene_frame_t frame);
    scene_plan_t finalize_open(boundary_audit_t boundary);
  };

  const char *boundary_decision_name(boundary_decision_e decision);
}  // namespace offline_sbs
