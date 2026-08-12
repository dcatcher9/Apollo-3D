/**
 * @file src/platform/windows/sbs_debug_dump_border.h
 * @brief Pure matched-frame validation for the optional Dump 3D window-region artifact.
 */
#pragma once

// standard includes
#include <cstdint>

namespace platf::sbs_debug {

  inline constexpr std::uint32_t window_region_schema = 1u;

  /** Semantic source that authorized an exact top-level-window analysis region. */
  enum class window_region_authority_kind_e : std::uint8_t {
    chromium_video,
    foreground_client,
  };

  [[nodiscard]] constexpr const char *window_region_authority_kind_name(
    const window_region_authority_kind_e kind
  ) noexcept {
    switch (kind) {
      case window_region_authority_kind_e::chromium_video:
        return "chromium-video";
      case window_region_authority_kind_e::foreground_client:
        return "foreground-client";
    }
    return "unknown";
  }

  /**
   * A capture-space, half-open window region stamped onto the same source frame later handed to
   * Dump 3D. The artifact remains provenance evidence; live crop/warp authority is separately
   * authenticated as a depth_input_region_t bound to the same matched frame.
   */
  struct window_region_snapshot {
    window_region_authority_kind_e authority_kind =
      window_region_authority_kind_e::chromium_video;
    std::uint64_t matched_frame_id = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::int32_t left = 0;
    std::int32_t top = 0;
    std::int32_t right = 0;
    std::int32_t bottom = 0;
    std::uint64_t hwnd = 0;
    std::uint32_t process_id = 0;
    std::int32_t document_id = 0;
    std::int32_t video_id = 0;
    std::uint64_t generation = 0;
    std::uint32_t latest_observation_age_ms_at_capture = 0;
    std::uint32_t maximum_observation_age_ms = 0;
    std::uint64_t geometry_continuity_ms_at_capture = 0;
    std::uint64_t source_content_age_ms_at_capture = 0;
  };

  enum class window_region_error : std::uint8_t {
    none,
    frame_mismatch,
    source_extent_mismatch,
    empty_or_out_of_bounds_rect,
    unknown_authority_kind,
    missing_identity,
    unexpected_dom_identity,
    stale,
    noncausal_geometry,
  };

  [[nodiscard]] constexpr window_region_error validate_window_region(
    const window_region_snapshot &region,
    const std::uint64_t matched_frame_id,
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) noexcept {
    if (region.matched_frame_id == 0 || region.matched_frame_id != matched_frame_id) {
      return window_region_error::frame_mismatch;
    }
    if (
      region.source_width == 0 || region.source_height == 0 ||
      region.source_width != source_width || region.source_height != source_height
    ) {
      return window_region_error::source_extent_mismatch;
    }
    if (
      region.left < 0 || region.top < 0 ||
      region.right <= region.left || region.bottom <= region.top ||
      static_cast<std::uint32_t>(region.right) > source_width ||
      static_cast<std::uint32_t>(region.bottom) > source_height
    ) {
      return window_region_error::empty_or_out_of_bounds_rect;
    }
    if (region.hwnd == 0 || region.process_id == 0 || region.generation == 0) {
      return window_region_error::missing_identity;
    }
    switch (region.authority_kind) {
      case window_region_authority_kind_e::chromium_video:
        if (region.document_id == 0 || region.video_id == 0) {
          return window_region_error::missing_identity;
        }
        break;
      case window_region_authority_kind_e::foreground_client:
        if (region.document_id != 0 || region.video_id != 0) {
          return window_region_error::unexpected_dom_identity;
        }
        break;
      default:
        return window_region_error::unknown_authority_kind;
    }
    if (
      region.maximum_observation_age_ms == 0 ||
      region.latest_observation_age_ms_at_capture > region.maximum_observation_age_ms
    ) {
      return window_region_error::stale;
    }
    if (region.geometry_continuity_ms_at_capture < region.source_content_age_ms_at_capture) {
      return window_region_error::noncausal_geometry;
    }
    return window_region_error::none;
  }

  [[nodiscard]] constexpr const char *window_region_error_name(
    const window_region_error error
  ) noexcept {
    switch (error) {
      case window_region_error::none:
        return "none";
      case window_region_error::frame_mismatch:
        return "frame-mismatch";
      case window_region_error::source_extent_mismatch:
        return "source-extent-mismatch";
      case window_region_error::empty_or_out_of_bounds_rect:
        return "empty-or-out-of-bounds-rect";
      case window_region_error::unknown_authority_kind:
        return "unknown-authority-kind";
      case window_region_error::missing_identity:
        return "missing-identity";
      case window_region_error::unexpected_dom_identity:
        return "unexpected-dom-identity";
      case window_region_error::stale:
        return "stale";
      case window_region_error::noncausal_geometry:
        return "noncausal-geometry";
    }
    return "unknown";
  }

}  // namespace platf::sbs_debug
