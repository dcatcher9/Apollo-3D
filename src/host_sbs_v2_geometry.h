/**
 * @file src/host_sbs_v2_geometry.h
 * @brief Shared CPU mirror of the Host SBS V2 pixel-shader geometry constants.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace models {

  /** Slot-for-slot CPU mirror of HostSbsV2Geometry at pixel-shader b2. */
  struct alignas(16) host_sbs_v2_geometry_t {
    float content_scale_x = 1.0f;
    float content_scale_y = 1.0f;
    float video_roi_active = 0.0f;
    float reserved = 0.0f;
    float video_roi_left = 0.0f;
    float video_roi_top = 0.0f;
    float video_roi_right = 1.0f;
    float video_roi_bottom = 1.0f;
    std::uint32_t tensor_content_left = 0u;
    std::uint32_t tensor_content_top = 0u;
    std::uint32_t tensor_content_right = 0u;
    std::uint32_t tensor_content_bottom = 0u;

    [[nodiscard]] constexpr bool operator==(const host_sbs_v2_geometry_t &) const = default;
  };

  /** Full-source renderer geometry used by every offline path and non-ROI live frames. */
  [[nodiscard]] constexpr host_sbs_v2_geometry_t
  make_host_sbs_v2_full_frame_geometry(
    const float content_scale_x,
    const float content_scale_y
  ) noexcept {
    host_sbs_v2_geometry_t geometry;
    geometry.content_scale_x = content_scale_x;
    geometry.content_scale_y = content_scale_y;
    return geometry;
  }

  static_assert(alignof(host_sbs_v2_geometry_t) == 16u);
  static_assert(sizeof(host_sbs_v2_geometry_t) == 48u);
  static_assert(offsetof(host_sbs_v2_geometry_t, content_scale_x) == 0u);
  static_assert(offsetof(host_sbs_v2_geometry_t, video_roi_active) == 8u);
  static_assert(offsetof(host_sbs_v2_geometry_t, video_roi_left) == 16u);
  static_assert(offsetof(host_sbs_v2_geometry_t, tensor_content_left) == 32u);

}  // namespace models
