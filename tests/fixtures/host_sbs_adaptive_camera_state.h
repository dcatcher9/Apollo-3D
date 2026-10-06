#pragma once

#include <src/depth_coordinate_v2.h>

namespace host_sbs_test {
  // Renderer-only fixtures need an authenticated camera but do not run the producer. The
  // synthetic clock belongs to this test fixture; production always uses real source timestamps.
  inline void seal_adaptive_camera(models::depth_coordinate_v2::state_words_t &words) {
    namespace v2 = models::depth_coordinate_v2;
    const float zero = std::bit_cast<float>(words[v2::center]);
    const float inverse = std::bit_cast<float>(words[v2::inverse_scale]);
    words[v2::joint_plane_mode_bits] = v2::adaptive_policy_id;
    std::fill(words.begin() + v2::gain_last_observation_low, words.end(), 0u);
    if (inverse > 0.0f) {
      words[v2::gain_last_observation_low] = 1u;
      words[v2::gain_clock_armed] = 1u;
      words[v2::gain_seed_count] = 1u;
      words[v2::gain_target_zero] = std::bit_cast<std::uint32_t>(zero);
      words[v2::gain_target_inverse_scale] = std::bit_cast<std::uint32_t>(inverse);
      words[v2::gain_target_nearest] = std::bit_cast<std::uint32_t>(1.0f / inverse);
      words[v2::gain_display_limit] = std::bit_cast<std::uint32_t>(v2::display_budget());
      words[v2::gain_seed_first_low] = words[v2::gain_seed_last_low] = 1u;
      words[v2::gain_seed_mean_nearest] = words[v2::gain_target_nearest];
      words[v2::gain_seed_mean_zero] = words[v2::gain_target_zero];
    }
    words[v2::camera_center_integrity_bits] = v2::camera_center_integrity_for_state_words(words);
  }
}
