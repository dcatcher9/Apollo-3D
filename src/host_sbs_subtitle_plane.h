/** @file Shared authentication of the current OCR/UI-plane publication. */
#pragma once

#include "depth_coordinate_v2.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace models::host_sbs_subtitle_plane {
  namespace v2 = depth_coordinate_v2;
  using words_t = std::array<std::uint32_t, v2::subtitle_plane_state_word_count>;
  using condition_words_t = std::array<std::uint32_t, v2::subtitle_condition_param_word_count>;
  // Source extent, field extent, analysis ROI, and content L/T/R/B, in wire order.
  using domain_words_t = std::array<std::uint32_t, 10u>;

  [[nodiscard]] constexpr std::uint64_t join(const std::uint32_t *words) noexcept {
    return words[0u] | (static_cast<std::uint64_t>(words[1u]) << 32u);
  }

  [[nodiscard]] constexpr std::uint32_t current_count(const words_t &words) noexcept {
    return std::popcount(words[v2::subtitle_plane_current_slot_mask_word]);
  }

  [[nodiscard]] inline bool state_words_are_canonical(
    const words_t &words, const float display_budget
  ) noexcept {
    const auto flags = words[v2::subtitle_plane_flags_word];
    const auto mask = words[v2::subtitle_plane_current_slot_mask_word];
    const auto resets = words[v2::subtitle_plane_history_reset_mask_word];
    const auto warmup = words[v2::subtitle_plane_warmup_word];
    constexpr auto warmup_known_bits = v2::subtitle_plane_warmup_remaining_mask | v2::subtitle_plane_warmup_cue_bit;
    constexpr auto capacity_mask = (1u << v2::subtitle_plane_cover_capacity) - 1u;
    if ((warmup & ~warmup_known_bits) != 0u ||
        (warmup & v2::subtitle_plane_warmup_remaining_mask) > v2::subtitle_plane_warmup_observations || (resets & ~mask) != 0u || words[0u] != v2::subtitle_plane_state_schema || words[1u] != v2::subtitle_plane_state_tag ||
        (flags & ~v2::subtitle_plane_current_valid_flag) != 0u ||
        (mask & ~capacity_mask) != 0u || (mask & (mask + 1u)) != 0u ||
        (flags == 0u && mask != 0u) ||
        !std::all_of(words.begin() + v2::subtitle_plane_reserved_offset,
          words.begin() + v2::subtitle_plane_adaptive_offset, [](auto value) { return value == 0u; })) {
      return false;
    }
    const auto *tail = words.data() + v2::subtitle_plane_adaptive_offset;
    if ((resets != 0u && tail[15u] != 0u) ||
        (warmup != 0u && (flags != v2::subtitle_plane_current_valid_flag || mask == 0u ||
          tail[0u] != 1u || tail[15u] != 0u))) return false;
    if (tail[0u] == 0u) {
      return mask == 0u && std::all_of(tail, tail + v2::subtitle_plane_adaptive_word_count,
        [](auto value) { return value == 0u; });
    }
    const auto source_width = words[v2::subtitle_plane_source_extent_offset];
    const auto *content = words.data() + v2::subtitle_plane_content_offset;
    if (source_width == 0u || content[0u] >= content[2u] || content[1u] >= content[3u] ||
        content[2u] > words[v2::subtitle_plane_field_extent_offset] ||
        content[3u] > words[v2::subtitle_plane_field_extent_offset + 1u]) {
      return false;
    }
    const auto area = static_cast<std::uint64_t>(content[2u] - content[0u]) * (content[3u] - content[1u]);
    const auto maximum_level = (static_cast<std::uint64_t>(source_width) + 99u) / 100u;
    const auto controller_flags = tail[1u];
    const auto applied = std::bit_cast<float>(tail[3u]);
    const auto clock = join(tail + 4u);
    const auto approach = join(tail + 6u);
    const auto retreat = join(tail + 8u);
    const auto covered = static_cast<std::uint64_t>(tail[13u]);
    const auto conflicts = static_cast<std::uint64_t>(tail[14u]);
    const bool capped = conflicts * 5u > covered || conflicts * 50u > area;
    return tail[0u] == 1u && (controller_flags & ~15u) == 0u && (controller_flags & 3u) != 3u &&
      tail[2u] <= maximum_level && tail[10u] <= maximum_level && tail[11u] <= maximum_level &&
      std::isfinite(display_budget) && display_budget > 0.0f &&
      tail[12u] == std::bit_cast<std::uint32_t>(display_budget) &&
      std::isfinite(applied) && applied >= 0.0f && applied <= display_budget &&
      covered <= area && conflicts <= covered && tail[15u] <= covered &&
      (((controller_flags & 8u) != 0u) == (clock != 0u)) &&
      (((controller_flags & 1u) != 0u) == (approach != 0u)) &&
      (((controller_flags & 2u) != 0u) == (retreat != 0u)) &&
      approach <= clock && retreat <= clock && (((controller_flags & 4u) != 0u) == capped);
  }

  [[nodiscard]] inline bool condition_words_match_plane(
    const condition_words_t &condition, const words_t &plane,
    const bool geometry_ready, const bool allow_unpublished = false
  ) noexcept {
    const auto count = current_count(plane);
    if (count == 0u && allow_unpublished && std::all_of(condition.begin(), condition.end(),
          [](auto value) { return value == 0u; })) {
      return true;
    }
    if (condition[0u] != v2::subtitle_condition_param_schema ||
        condition[1u] != v2::subtitle_condition_param_tag) return false;
    // A current cover union can contain an invalid Base probe. Retain its OCR identity,
    // but do not grant conditioning or temporal confirmation from that observation.
    if (count == 0u || plane[v2::subtitle_plane_adaptive_offset + 15u] != 0u) {
      return std::all_of(condition.begin() + 2u, condition.end(), [](auto value) { return value == 0u; });
    }
    return geometry_ready && plane[v2::subtitle_plane_flags_word] == v2::subtitle_plane_current_valid_flag &&
      condition[v2::subtitle_condition_param_current_count_word] == count &&
      (condition[v2::subtitle_condition_param_current_kind_mask_word] & ~((1u << count) - 1u)) == 0u &&
      condition[v2::subtitle_condition_param_active_word] == v2::subtitle_condition_param_fully_active_value &&
      condition[v2::subtitle_condition_param_target_word] == plane[v2::subtitle_plane_adaptive_offset + 3u];
  }
}  // namespace models::host_sbs_subtitle_plane
