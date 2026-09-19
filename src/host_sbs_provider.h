#pragma once

#include <cstdint>

namespace video {
  /** Final SBS is already authored: neither eye may be resized to fit an encoder cap. */
  [[nodiscard]] constexpr bool external_sbs_dimensions_match(
    int source_width,
    int source_height,
    int packed_width,
    int packed_height
  ) noexcept {
    return source_width >= 2 && source_height >= 2 && source_width % 2 == 0 &&
           source_height % 2 == 0 &&
           static_cast<std::int64_t>(source_width) * 2 == packed_width &&
           source_height == packed_height;
  }
}  // namespace video
