/**
 * @file src/sbs_roi_tracker.h
 * @brief Deterministic candidate selection and temporal tracking for Host SBS content ROIs.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace sbs_roi {
  enum class roi_kind_e {
    full_frame,
    video,
    content,
    scroll_hold,
  };

  /**
   * Normalized half-open source rectangle. Valid coordinates satisfy
   * 0 <= left < right <= 1 and 0 <= top < bottom <= 1.
   */
  struct normalized_rect_t {
    float left = 0.0f;
    float top = 0.0f;
    float right = 1.0f;
    float bottom = 1.0f;
  };

  /**
   * One rectangle proposed by the low-resolution feature extractor.
   *
   * The tracker deliberately receives evidence rather than pixels so the same
   * state machine can be driven by deterministic unit tests and the future
   * bounded D3D11 readback path.
   */
  struct candidate_t {
    roi_kind_e kind = roi_kind_e::content;
    normalized_rect_t rect;
    float temporal_occupancy = 0.0f;
    float photographic_density = 0.0f;
    float primary_column_support = 0.0f;
    float gutter_confidence = 0.0f;
    bool inside_primary_column = false;
    bool crosses_stable_gutter = false;
    bool recent_interaction = false;
  };

  struct observation_t {
    std::uint64_t id = 0;  ///< Strictly increasing detector/source identity.
    std::uint64_t timestamp_us = 0;  ///< Strictly increasing capture timestamp.
    /// Trustworthy local monotonic receipt time used for dwell and health timeouts.
    std::uint64_t arrival_timestamp_us = 0;
    std::vector<candidate_t> candidates;
    /// True only for spatially broad page translation, never a pan inside the player.
    bool broad_page_scroll = false;
    /// Monotonic one-shot display/HDR/layout invalidation event; zero means no event.
    std::uint64_t invalidation_epoch = 0;
  };

  struct tracker_config_t {
    std::size_t max_candidates = 64;
    std::uint32_t acquire_updates = 3;
    std::uint32_t challenger_updates = 5;
    std::uint32_t release_updates = 600;
    std::uint32_t scroll_enter_updates = 2;
    std::uint32_t scroll_exit_updates = 3;
    std::uint32_t scroll_reacquire_timeout_updates = 15;
    std::uint32_t invalid_input_timeout_updates = 3;
    std::uint32_t weak_retention_grace_updates = 30;

    std::uint64_t max_continuity_gap_us = 350000;
    std::uint64_t acquire_duration_us = 120000;
    std::uint64_t challenger_duration_us = 300000;
    std::uint64_t release_duration_us = 60000000;
    std::uint64_t scroll_enter_duration_us = 50000;
    std::uint64_t scroll_exit_duration_us = 120000;
    std::uint64_t scroll_reacquire_timeout_us = 1200000;
    std::uint64_t invalid_input_timeout_us = 200000;
    std::uint64_t weak_retention_grace_us = 2000000;

    float association_iou = 0.60f;
    float stable_geometry_iou = 0.92f;
    float min_video_area = 0.12f;
    float min_content_area = 0.08f;
    float min_video_temporal_occupancy = 0.08f;
    float min_video_photographic_density = 0.08f;
    float min_content_photographic_density = 0.18f;
    float min_retention_evidence = 0.02f;
    float min_primary_column_support = 0.50f;
    float min_gutter_confidence = 0.35f;
    float video_winner_ratio = 1.60f;
    float content_winner_ratio = 1.20f;
    float challenger_ratio = 1.35f;
    float interaction_min_score_ratio = 0.85f;
    float identity_area = 0.90f;
    float identity_edge_margin = 0.05f;
  };

  enum class update_status_e {
    accepted,
    ignored_stale,
    ignored_invalid,
    invalidated_invalid_input,
  };

  struct tracker_output_t {
    roi_kind_e kind = roi_kind_e::full_frame;
    normalized_rect_t rect;
    float confidence = 0.0f;
    std::uint64_t generation = 0;
    std::uint64_t observation_id = 0;
    std::uint64_t committed_observation_id = 0;
    std::uint32_t age_updates = 0;
    bool committed_this_update = false;
    update_status_e status = update_status_e::accepted;
  };

  class tracker_t {
  public:
    explicit tracker_t(tracker_config_t config = {});

    /**
     * Consumes one distinct detector observation.
     *
     * Invalid, duplicate, and out-of-order observations are ignored without
     * mutating tracker state. Candidate exploration never changes the committed
     * rectangle or generation.
     */
    tracker_output_t update(const observation_t &observation);

    /**
     * Advances the detector-health watchdog when no trustworthy observation
     * envelope is available. `arrival_timestamp_us` must come from the caller's
     * local monotonic clock, not the detector payload.
     */
    tracker_output_t detector_unavailable(std::uint64_t arrival_timestamp_us);

    [[nodiscard]] tracker_output_t snapshot() const;

  private:
    struct scored_candidate_t {
      candidate_t candidate;
      float score = 0.0f;
    };

    struct tracked_candidate_t {
      scored_candidate_t value;
      normalized_rect_t anchor_rect;
      std::uint32_t updates = 0;
      std::uint64_t first_timestamp_us = 0;
    };

    tracker_config_t config_;
    roi_kind_e committed_kind_ = roi_kind_e::full_frame;
    normalized_rect_t committed_rect_;
    float committed_confidence_ = 0.0f;
    std::uint64_t generation_ = 0;
    std::uint64_t last_observation_id_ = 0;
    std::uint64_t last_timestamp_us_ = 0;
    std::uint64_t last_arrival_timestamp_us_ = 0;
    std::uint64_t last_invalidation_epoch_ = 0;
    std::uint64_t committed_observation_id_ = 0;
    std::uint32_t committed_age_updates_ = 0;
    std::uint32_t missing_updates_ = 0;
    std::uint32_t invalid_updates_ = 0;
    std::uint32_t weak_retention_updates_ = 0;
    std::uint32_t scroll_updates_ = 0;
    std::uint32_t rest_updates_ = 0;
    std::uint32_t scroll_reacquire_updates_ = 0;
    std::optional<std::uint64_t> missing_since_us_;
    std::optional<std::uint64_t> invalid_since_us_;
    std::optional<std::uint64_t> weak_retention_since_us_;
    std::optional<std::uint64_t> scroll_since_us_;
    std::optional<std::uint64_t> rest_since_us_;
    std::optional<std::uint64_t> scroll_reacquire_since_us_;
    bool scroll_hold_ = false;
    bool scroll_reacquiring_ = false;
    bool invalid_fail_safe_active_ = false;
    std::optional<tracked_candidate_t> acquisition_;
    std::optional<tracked_candidate_t> challenger_;

    [[nodiscard]] bool valid_candidates(const observation_t &observation) const;
    [[nodiscard]] std::optional<scored_candidate_t> select_candidate(const observation_t &observation) const;
    [[nodiscard]] std::optional<scored_candidate_t> find_incumbent(const observation_t &observation) const;
    void track_acquisition(const scored_candidate_t &candidate, std::uint64_t timestamp_us);
    void track_challenger(const scored_candidate_t &candidate, std::uint64_t timestamp_us);
    [[nodiscard]] bool acquisition_mature(std::uint64_t timestamp_us) const;
    [[nodiscard]] bool challenger_mature(std::uint64_t timestamp_us) const;
    [[nodiscard]] tracker_output_t handle_invalid_input(
      std::uint64_t arrival_timestamp_us,
      update_status_e status
    );
    void clear_search_state();
    void clear_scroll_state();
    [[nodiscard]] bool commit(const scored_candidate_t &candidate, std::uint64_t observation_id);
    [[nodiscard]] bool commit_full_frame(std::uint64_t observation_id, bool force_generation);
    [[nodiscard]] tracker_output_t output(
      update_status_e status = update_status_e::accepted,
      bool committed_this_update = false
    ) const;
  };
}  // namespace sbs_roi
