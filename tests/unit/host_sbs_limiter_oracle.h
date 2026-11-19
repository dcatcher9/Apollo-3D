#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <src/depth_coordinate_v2.h>
#include <vector>

namespace host_sbs_test {
  inline float vertical_envelope_share(const float majorant, const float minorant) {
    const float majorant_share = models::depth_coordinate_v2::vertical_majorant_share;
    volatile float majorant_term = majorant_share * majorant;
    volatile float minorant_term = (1.0f - majorant_share) * minorant;
    return majorant_term + minorant_term;
  }

  struct q30_vertical_oracle_t {
    std::vector<float> majorant;
    std::vector<float> conditioned;
  };

  inline std::int32_t limiter_upper_q30(float value) {
    namespace v2 = models::depth_coordinate_v2;
    value = std::isfinite(value) ?
      std::clamp(value, -v2::direct_container_limit, v2::direct_container_limit) :
      0.0f;
    return static_cast<std::int32_t>(
      std::ceil(value * static_cast<float>(v2::limiter_q_scale))
    );
  }

  inline std::int32_t limiter_lower_q30(float value) {
    namespace v2 = models::depth_coordinate_v2;
    value = std::isfinite(value) ?
      std::clamp(value, -v2::direct_container_limit, v2::direct_container_limit) :
      0.0f;
    return static_cast<std::int32_t>(
      std::floor(value * static_cast<float>(v2::limiter_q_scale))
    );
  }

  inline float limiter_from_q30(const std::int32_t value) {
    return static_cast<float>(value) /
           static_cast<float>(models::depth_coordinate_v2::limiter_q_scale);
  }

  inline std::int32_t limiter_step_q30(
    const std::uint32_t numerator,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (content_width == 0u) {
      return 0;
    }
    const std::uint32_t max_decay =
      2u * static_cast<std::uint32_t>(v2::limiter_container_q_limit);
    return static_cast<std::int32_t>(std::max(
      1u,
      std::min(numerator / content_width, max_decay)
    ));
  }

  inline q30_vertical_oracle_t exact_q30_vertical_oracle(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (width == 0u || height <= v2::limiter_serial_max_lines || content_width == 0u ||
        candidate.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }

    std::vector<std::int32_t> upper(candidate.size());
    std::vector<std::int32_t> lower(candidate.size());
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      upper[index] = limiter_upper_q30(candidate[index]);
      lower[index] = limiter_lower_q30(candidate[index]);
    }
    const std::int32_t step = limiter_step_q30(
      v2::limiter_vertical_step_q_numerator,
      content_width
    );
    for (std::uint32_t x = 0u; x < width; ++x) {
      for (std::uint32_t y = 1u; y < height; ++y) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        upper[index] = std::max(upper[index], upper[index - width] - step);
        lower[index] = std::min(lower[index], lower[index - width] + step);
      }
      for (std::uint32_t y = height - 1u; y > 0u; --y) {
        const std::size_t index = static_cast<std::size_t>(y - 1u) * width + x;
        upper[index] = std::max(upper[index], upper[index + width] - step);
        lower[index] = std::min(lower[index], lower[index + width] + step);
      }
    }

    q30_vertical_oracle_t result;
    result.majorant.resize(candidate.size());
    result.conditioned.resize(candidate.size());
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      const float majorant = limiter_from_q30(upper[index]);
      const float minorant = limiter_from_q30(lower[index]);
      result.majorant[index] = majorant;
      result.conditioned[index] = std::clamp(
        vertical_envelope_share(majorant, minorant),
        minorant,
        majorant
      );
    }
    return result;
  }

  inline std::vector<float> exact_q30_horizontal_oracle(
    const std::vector<float> &candidate,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t content_width
  ) {
    namespace v2 = models::depth_coordinate_v2;
    if (width <= v2::limiter_serial_max_lines || height == 0u || content_width == 0u ||
        candidate.size() != static_cast<std::size_t>(width) * height) {
      return {};
    }

    std::vector<std::int32_t> result_q30(candidate.size());
    std::transform(
      candidate.begin(),
      candidate.end(),
      result_q30.begin(),
      limiter_upper_q30
    );
    const std::int32_t step = limiter_step_q30(
      v2::limiter_horizontal_step_q_numerator,
      content_width
    );
    for (std::uint32_t y = 0u; y < height; ++y) {
      const std::size_t row = static_cast<std::size_t>(y) * width;
      for (std::uint32_t x = 1u; x < width; ++x) {
        const std::size_t index = row + x;
        result_q30[index] = std::max(
          result_q30[index],
          result_q30[index - 1u] - step
        );
      }
      for (std::uint32_t x = width - 1u; x > 0u; --x) {
        const std::size_t index = row + x - 1u;
        result_q30[index] = std::max(
          result_q30[index],
          result_q30[index + 1u] - step
        );
      }
    }

    std::vector<float> result(candidate.size());
    std::transform(
      result_q30.begin(),
      result_q30.end(),
      result.begin(),
      limiter_from_q30
    );
    return result;
  }
}  // namespace host_sbs_test
