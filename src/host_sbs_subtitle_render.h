/** @file Exact source/permission constants for the native glyph compositor. */
#pragma once

#include <cstddef>
#include <cstdint>

namespace models {
  /** Slot-for-slot mirror of HostSbsSubtitleRender at pixel-shader b3. */
  struct alignas(16) host_sbs_subtitle_render_constants_t {
    std::uint32_t enabled = 0u;
    std::uint32_t color_mode = 0u;
    std::uint32_t analysis_generation_low = 0u;
    std::uint32_t analysis_generation_high = 0u;
    std::uint32_t source_left = 0u;
    std::uint32_t source_top = 0u;
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;

    [[nodiscard]] constexpr bool operator==(const host_sbs_subtitle_render_constants_t &) const = default;
  };

  static_assert(sizeof(host_sbs_subtitle_render_constants_t) == 32u);
  static_assert(alignof(host_sbs_subtitle_render_constants_t) == 16u);
  static_assert(offsetof(host_sbs_subtitle_render_constants_t, source_left) == 16u);
}  // namespace models
