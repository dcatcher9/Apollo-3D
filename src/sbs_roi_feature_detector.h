/**
 * @file src/sbs_roi_feature_detector.h
 * @brief Bounded feature-grid candidate extraction for Host SBS content ROIs.
 */
#pragma once

#include "sbs_roi_tracker.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sbs_roi {
  inline constexpr std::uint16_t nominal_feature_grid_width = 128;
  inline constexpr std::uint16_t nominal_feature_grid_height = 72;

  /**
   * One quantized cell produced by the future nonblocking D3D11 analysis pass.
   *
   * `temporal_occupancy_q8` is exposure-invariant changed-pixel occupancy, not
   * color-difference magnitude. `vertical_shift_rows` is signed displacement in
   * analysis-grid rows.
   */
  struct feature_cell_t {
    std::uint8_t temporal_occupancy_q8 = 0;
    std::uint8_t photographic_density_q8 = 0;
    std::uint8_t gutter_stability_q8 = 0;
    std::int8_t vertical_shift_rows = 0;
    std::uint8_t vertical_shift_confidence_q8 = 0;
  };

  struct feature_grid_view_t {
    std::uint16_t width = nominal_feature_grid_width;
    std::uint16_t height = nominal_feature_grid_height;
    std::span<const feature_cell_t> cells;
  };

  struct normalized_point_t {
    float x = 0.0f;
    float y = 0.0f;
  };

  /**
   * A known video motion region is excluded from page-scroll voting so an
   * in-player camera pan cannot authoritatively put the ROI tracker into
   * scroll hold. The exclusion is independent of render kind because a
   * fullscreen video renders as the identity/full-frame ROI.
   */
  struct feature_detector_context_t {
    std::optional<normalized_rect_t> video_motion_exclusion;
  };

  struct feature_detector_config_t {
    std::size_t max_cells = 32768;
    std::size_t max_candidates = 64;
    std::size_t max_components = 256;
    std::uint16_t max_grid_dimension = 512;

    std::uint8_t video_activity_threshold_q8 = 40;
    std::uint8_t video_photo_threshold_q8 = 24;
    std::uint8_t content_photo_threshold_q8 = 64;
    std::uint8_t gutter_stability_threshold_q8 = 176;
    std::uint8_t scroll_confidence_threshold_q8 = 160;

    std::uint16_t min_gutter_width_cells = 3;
    std::uint16_t video_bridge_x_cells = 2;
    std::uint16_t video_bridge_y_cells = 2;
    std::uint16_t content_bridge_x_cells = 5;
    std::uint16_t content_bridge_y_cells = 5;
    std::uint16_t content_halo_cells = 1;
    std::uint16_t min_component_width_cells = 3;
    std::uint16_t min_component_height_cells = 3;

    float gutter_row_fraction = 0.76f;
    float primary_column_width_ratio = 1.20f;
    float content_trim_fraction = 0.05f;
    float content_trim_min_density_gain = 1.08f;
    float content_min_area = 0.04f;
    float content_min_width = 0.20f;
    float content_min_height = 0.16f;
    float content_min_fill = 0.12f;
    float content_axis_fill = 0.25f;
    float content_min_axis_occupancy = 0.45f;
    float content_max_aspect_ratio = 3.50f;
    float structural_competitor_min_area = 0.03f;
    float no_gutter_video_min_content_ratio = 0.85f;
    float scroll_min_support_fraction = 0.24f;
    float scroll_min_direction_agreement = 0.82f;
    float scroll_min_band_support_fraction = 0.08f;
    std::int8_t scroll_min_shift_rows = 1;
  };

  enum class feature_detection_status_e {
    accepted,
    invalid_shape,
    invalid_interaction,
    invalid_context,
    overflow,
  };

  struct feature_gutter_t {
    normalized_rect_t rect;
    float confidence = 0.0f;
  };

  struct feature_column_t {
    normalized_rect_t rect;
    float evidence = 0.0f;
    bool primary = false;
  };

  struct feature_detection_result_t {
    feature_detection_status_e status = feature_detection_status_e::accepted;
    std::vector<candidate_t> candidates;
    std::vector<feature_gutter_t> gutters;
    std::vector<feature_column_t> columns;
    std::optional<normalized_rect_t> primary_column;
    bool primary_column_ambiguous = false;
    bool broad_page_scroll = false;
    float vertical_scroll_rows = 0.0f;
    float scroll_confidence = 0.0f;
  };

  /**
   * Pure feature-map geometry detector.
   *
   * It owns no frame identity, timing, generation, or tracker state. Callers
   * must treat any non-accepted result as detector unavailability rather than
   * as an accepted observation containing no candidates.
   */
  class feature_detector_t {
  public:
    explicit feature_detector_t(feature_detector_config_t config = {});

    [[nodiscard]] feature_detection_result_t detect(
      feature_grid_view_t grid,
      std::optional<normalized_point_t> recent_interaction = std::nullopt,
      feature_detector_context_t context = {}
    ) const;

  private:
    feature_detector_config_t config_;
  };
}  // namespace sbs_roi
