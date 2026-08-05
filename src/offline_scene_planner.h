/**
 * @file src/offline_scene_planner.h
 * @brief Bounded-lookahead scene planning for native offline SBS conversion.
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
  inline constexpr std::size_t default_max_open_scene_frames = 524288;
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
   * `sequence` is one-based and contiguous. Metrics which were not exported must
   * remain std::nullopt; a missing diagnostic is deliberately different from a
   * measured zero. The structureless analysis bit alone does not distinguish
   * the first held low-structure update from its persistent successor. The
   * planner therefore recognizes the live structureless geometry exception
   * only when the producer's geometry-candidate bit or causal hard-cut pulse is
   * also present. A trace missing those diagnostics can retain its own proposal
   * conservatively, but cannot prove or relocate that held endpoint offline.
   */
  struct scene_frame_t {
    std::uint64_t sequence = 0;
    std::string frame_id;
    bool depth_updated = false;
    bool hard_cut_pulse = false;
    std::optional<float> depth_change_fraction;
    std::optional<float> raw_rgb_change_fraction;
    std::optional<float> structural_change_fraction;
    std::optional<float> current_structural_support_fraction;
    std::optional<float> previous_structural_support_fraction;
    std::optional<float> common_structural_support_fraction;
    std::uint32_t analysis_flags = 0;
    std::optional<double> pts_seconds;
    std::optional<double> duration_seconds;
    std::uint64_t cache_bytes = 0;
  };

  struct scene_planner_config_t {
    std::size_t lookbehind_depth_updates = 4;
    std::size_t lookahead_depth_updates = 8;
    std::size_t duplicate_pulse_distance_updates = 2;
    std::size_t minimum_scene_frames = 2;
    std::uint64_t max_open_cache_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    std::size_t max_open_frames = default_max_open_scene_frames;
    bool allow_administrative_split = false;
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
    std::string cut_state_semantics = "causal-production-unrevised";
    std::string known_limit =
      "offline lookahead revises cache/replay boundaries only; V2 geometry remains unchanged";
  };

  class scene_plan_error: public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
  };

  class scene_cache_budget_error: public scene_plan_error {
  public:
    scene_cache_budget_error(
      std::uint64_t limit_bytes,
      std::uint64_t live_bytes,
      std::uint64_t open_start_sequence,
      std::uint64_t current_sequence
    );

    std::uint64_t limit_bytes;
    std::uint64_t live_bytes;
    std::uint64_t open_start_sequence;
    std::uint64_t current_sequence;
  };

  class scene_metadata_budget_error: public scene_plan_error {
  public:
    scene_metadata_budget_error(
      std::size_t limit_frames,
      std::size_t attempted_frames,
      std::uint64_t open_start_sequence,
      std::uint64_t current_sequence
    );

    std::size_t limit_frames;
    std::size_t attempted_frames;
    std::uint64_t open_start_sequence;
    std::uint64_t current_sequence;
  };

  /**
   * Delays each causal cut proposal until bounded future depth evidence arrives,
   * then emits immutable cache/replay boundaries. It never labels its
   * decisions as ground truth and retains only the unresolved scene.
   */
  class scene_planner_t {
  public:
    explicit scene_planner_t(scene_planner_config_t config);
    ~scene_planner_t();

    scene_planner_t(const scene_planner_t &) = delete;
    scene_planner_t &operator=(const scene_planner_t &) = delete;
    scene_planner_t(scene_planner_t &&) = delete;
    scene_planner_t &operator=(scene_planner_t &&) = delete;

    std::vector<scene_plan_t> feed(scene_frame_t frame);
    std::vector<scene_plan_t> finish();

    [[nodiscard]] std::uint64_t open_cache_bytes() const;
    [[nodiscard]] std::size_t open_frame_count() const;
    [[nodiscard]] std::uint64_t open_start_sequence() const;
    [[nodiscard]] std::size_t pending_proposal_count() const;
    [[nodiscard]] const std::vector<boundary_audit_t> &boundary_audit() const;

  private:
    struct tracked_frame_t;
    struct proposal_cluster_t;

    scene_planner_config_t config_;
    std::vector<tracked_frame_t> frames_;
    std::vector<proposal_cluster_t> pending_;
    std::vector<boundary_audit_t> boundary_audit_;
    std::uint64_t next_sequence_ = 1;
    std::int64_t depth_update_ordinal_ = -1;
    std::optional<double> timeline_origin_seconds_;
    std::optional<double> last_depth_pts_seconds_;
    std::uint64_t scene_number_ = 0;
    std::uint64_t semantic_scene_number_ = 1;
    std::uint64_t open_cache_bytes_ = 0;
    bool closed_ = false;

    void add_proposal(const tracked_frame_t &frame);
    std::vector<scene_plan_t> resolve_mature(bool eof);
    std::optional<scene_plan_t> resolve_cluster(
      const proposal_cluster_t &cluster,
      bool truncated,
      bool budget_forced = false
    );
    std::vector<scene_plan_t> enforce_budget();
    scene_plan_t finalize_prefix(
      std::size_t prefix_count,
      boundary_audit_t boundary,
      bool increment_semantic_scene = true
    );
  };

  const char *boundary_decision_name(boundary_decision_e decision);
}  // namespace offline_sbs
