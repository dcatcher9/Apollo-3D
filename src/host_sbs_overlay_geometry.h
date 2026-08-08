#pragma once

#include "host_sbs_resolution.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace models {

  /** Maximum number of disjoint burned-in overlay regions authorized for one source frame. */
  inline constexpr std::size_t host_sbs_overlay_max_loose_rects = 8u;

  /** CPU mirror of OverlayZeroPlaneConstants at compute-shader register b2. */
  struct alignas(16) host_sbs_overlay_zero_plane_constants_t {
    std::uint32_t analysis_width = 0u;
    std::uint32_t analysis_height = 0u;
    std::uint32_t loose_rect_count = 0u;
    std::uint32_t reserved = 0u;
  };

  /**
   * Fixed-capacity, allocation-free geometry payload for one authenticated overlay plan.
   *
   * Rectangles are half-open pixels in the current analysis domain (full source or video crop),
   * not model-tensor coordinates. The detector owns stroke-to-loose-region construction; the
   * final-field consumer owns the additional one-texel sampling guard and slope-safe exterior
   * collar. Every active rectangle targets the common zero-parallax screen plane.
   */
  struct host_sbs_overlay_geometry_t {
    std::array<depth_source_rect_t, host_sbs_overlay_max_loose_rects>
      loose_rects {};
    std::uint32_t loose_rect_count = 0u;

    constexpr bool empty() const noexcept {
      return loose_rect_count == 0u;
    }

    constexpr bool valid(
      const std::uint32_t analysis_width,
      const std::uint32_t analysis_height
    ) const noexcept {
      if (analysis_width == 0u || analysis_height == 0u ||
          loose_rect_count > loose_rects.size()) {
        return false;
      }

      for (std::size_t index = 0u; index < loose_rects.size(); ++index) {
        const auto &rect = loose_rects[index];
        if (index < loose_rect_count) {
          if (!rect.valid() || rect.right > analysis_width ||
              rect.bottom > analysis_height) {
            return false;
          }
        } else if (rect != depth_source_rect_t {}) {
          // Canonical unused slots prevent stale rectangles from surviving a count decrease.
          return false;
        }
      }
      return true;
    }

    constexpr bool operator==(const host_sbs_overlay_geometry_t &) const = default;
  };

  static_assert(std::is_standard_layout_v<host_sbs_overlay_geometry_t>);
  static_assert(std::is_trivially_copyable_v<host_sbs_overlay_geometry_t>);
  static_assert(sizeof(depth_source_rect_t) == 4u * sizeof(std::uint32_t));
  static_assert(std::is_standard_layout_v<depth_source_rect_t>);
  static_assert(std::is_trivially_copyable_v<depth_source_rect_t>);
  static_assert(sizeof(host_sbs_overlay_zero_plane_constants_t) == 16u);
  static_assert(std::is_standard_layout_v<host_sbs_overlay_zero_plane_constants_t>);
  static_assert(std::is_trivially_copyable_v<host_sbs_overlay_zero_plane_constants_t>);

}  // namespace models
