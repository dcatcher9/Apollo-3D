/**
 * @file src/platform/windows/sbs_debug_dump_border.h
 * @brief Pure matched-frame validation for the optional Dump 3D window-video border artifact.
 */
#pragma once

// standard includes
#include <cstdint>

namespace platf::sbs_debug {

  inline constexpr std::uint32_t window_video_border_schema = 2u;

  /**
   * A capture-space, half-open video rectangle stamped onto the same source frame later handed to
   * Dump 3D. This is diagnostic evidence only; it never authorizes a crop, changes convergence,
   * or alters live Host-SBS geometry.
   */
  struct window_video_border_snapshot {
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
    std::uint32_t latest_heartbeat_age_ms_at_capture = 0;
    std::uint32_t maximum_heartbeat_age_ms = 0;
    std::uint64_t geometry_continuity_ms_at_capture = 0;
    std::uint64_t source_content_age_ms_at_capture = 0;
  };

  enum class window_video_border_error : std::uint8_t {
    none,
    frame_mismatch,
    source_extent_mismatch,
    empty_or_out_of_bounds_rect,
    missing_identity,
    stale,
    noncausal_geometry,
  };

  [[nodiscard]] constexpr window_video_border_error validate_window_video_border(
    const window_video_border_snapshot &border,
    const std::uint64_t matched_frame_id,
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) noexcept {
    if (border.matched_frame_id == 0 || border.matched_frame_id != matched_frame_id) {
      return window_video_border_error::frame_mismatch;
    }
    if (
      border.source_width == 0 || border.source_height == 0 ||
      border.source_width != source_width || border.source_height != source_height
    ) {
      return window_video_border_error::source_extent_mismatch;
    }
    if (
      border.left < 0 || border.top < 0 ||
      border.right <= border.left || border.bottom <= border.top ||
      static_cast<std::uint32_t>(border.right) > source_width ||
      static_cast<std::uint32_t>(border.bottom) > source_height
    ) {
      return window_video_border_error::empty_or_out_of_bounds_rect;
    }
    if (
      border.hwnd == 0 || border.process_id == 0 ||
      border.document_id == 0 || border.video_id == 0 || border.generation == 0
    ) {
      return window_video_border_error::missing_identity;
    }
    if (
      border.maximum_heartbeat_age_ms == 0 ||
      border.latest_heartbeat_age_ms_at_capture > border.maximum_heartbeat_age_ms
    ) {
      return window_video_border_error::stale;
    }
    if (border.geometry_continuity_ms_at_capture < border.source_content_age_ms_at_capture) {
      return window_video_border_error::noncausal_geometry;
    }
    return window_video_border_error::none;
  }

  [[nodiscard]] constexpr const char *window_video_border_error_name(
    const window_video_border_error error
  ) noexcept {
    switch (error) {
      case window_video_border_error::none:
        return "none";
      case window_video_border_error::frame_mismatch:
        return "frame-mismatch";
      case window_video_border_error::source_extent_mismatch:
        return "source-extent-mismatch";
      case window_video_border_error::empty_or_out_of_bounds_rect:
        return "empty-or-out-of-bounds-rect";
      case window_video_border_error::missing_identity:
        return "missing-identity";
      case window_video_border_error::stale:
        return "stale";
      case window_video_border_error::noncausal_geometry:
        return "noncausal-geometry";
    }
    return "unknown";
  }

}  // namespace platf::sbs_debug
