/**
 * @file src/sbs_scene_cache_contract.h
 * @brief Binary frame contract shared by the offline Host SBS cache producer and consumer.
 *
 * Cache schema 2 deliberately keeps depth payloads variable-sized. Every frame therefore owns a
 * fixed metadata record containing its exact depth dimensions and retained GPU ROI transform.
 * The state record also retains DepthFrameState so replay can reproduce the live
 * preserve-previous-on-invalid draw instead of rendering an invalid completion as a new frame.
 */
#pragma once

#include "generated/sbs_adaptive_state_contract.h"
#include "sbs_frame_roi_transform.h"
#include "sbs_roi_shape_request.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sbs_scene_cache {
  inline constexpr std::uint32_t contract_schema = 2u;
  inline constexpr std::uint32_t cached_state_schema = 3u;
  inline constexpr std::uint32_t frame_metadata_magic = 0x32434253u;  // "SBC2"
  inline constexpr std::uint32_t frame_metadata_schema = 1u;
  inline constexpr std::uint32_t frame_metadata_vector_count = 12u;
  inline constexpr std::uint32_t frame_metadata_word_count =
    frame_metadata_vector_count * 4u;
  inline constexpr std::uint32_t roi_transform_word_count =
    models::frame_roi_transform_vector_count * 4u;
  inline constexpr std::uint32_t roi_transform_word_offset = 16u;
  inline constexpr std::uint64_t unbound_source_frame_id =
    std::numeric_limits<std::uint64_t>::max();
  inline constexpr std::uint32_t max_depth_dimension =
    models::sbs_roi_shape_request_engine_max_dimension;
  static_assert(
    models::frame_roi_model_patch_size ==
      models::sbs_roi_shape_request_patch_size
  );

  inline constexpr std::size_t depth_frame_state_word_count = 4u;
  inline constexpr std::size_t cached_state_word_count =
    sbs_adaptive_state::render_prefix_word_count +
    depth_frame_state_word_count;

  using roi_transform_words_t =
    std::array<std::uint32_t, roi_transform_word_count>;
  using cached_state_words_t =
    std::array<std::uint32_t, cached_state_word_count>;

  /**
   * Exact 192-byte little-endian per-frame descriptor.
   *
   * [0] header:     magic, metadata schema, word count, ROI-transform word offset
   * [1] depth:      width, height, pixel count, bytes per sample
   * [2] sequence:   cache sequence low/high, retained source-frame id low/high
   * [3] identity:   source width/height, model width/height
   * [4..11] exact StructuredBuffer<uint4> FrameRoiTransform contents
   */
  struct alignas(16) frame_metadata_t {
    std::array<std::uint32_t, 4> header {
      frame_metadata_magic,
      frame_metadata_schema,
      frame_metadata_word_count,
      roi_transform_word_offset,
    };
    std::array<std::uint32_t, 4> depth {};
    std::array<std::uint32_t, 4> sequence {};
    std::array<std::uint32_t, 4> identity {};
    roi_transform_words_t roi_transform {};
  };

  static_assert(sizeof(frame_metadata_t) == frame_metadata_word_count * 4u);
  static_assert(alignof(frame_metadata_t) == 16u);
  static_assert(std::is_standard_layout_v<frame_metadata_t>);
  static_assert(std::is_trivially_copyable_v<frame_metadata_t>);
  static_assert(offsetof(frame_metadata_t, header) == 0u);
  static_assert(offsetof(frame_metadata_t, depth) == 16u);
  static_assert(offsetof(frame_metadata_t, sequence) == 32u);
  static_assert(offsetof(frame_metadata_t, identity) == 48u);
  static_assert(
    offsetof(frame_metadata_t, roi_transform) ==
      roi_transform_word_offset * sizeof(std::uint32_t)
  );

  [[nodiscard]] constexpr std::uint64_t join_u64(
    const std::uint32_t low,
    const std::uint32_t high
  ) {
    return static_cast<std::uint64_t>(low) |
           (static_cast<std::uint64_t>(high) << 32u);
  }

  constexpr void split_u64(
    const std::uint64_t value,
    std::uint32_t &low,
    std::uint32_t &high
  ) {
    low = static_cast<std::uint32_t>(value);
    high = static_cast<std::uint32_t>(value >> 32u);
  }

  [[nodiscard]] constexpr std::uint64_t cache_sequence(
    const frame_metadata_t &value
  ) {
    return join_u64(value.sequence[0], value.sequence[1]);
  }

  [[nodiscard]] constexpr std::uint64_t retained_source_frame_id(
    const frame_metadata_t &value
  ) {
    return join_u64(value.sequence[2], value.sequence[3]);
  }

  [[nodiscard]] constexpr bool transform_is_unbound_zero(
    const roi_transform_words_t &words
  ) {
    return std::all_of(words.begin(), words.end(), [](const auto word) {
      return word == 0u;
    });
  }

  [[nodiscard]] inline bool finite_rect(
    const roi_transform_words_t &words,
    const std::size_t offset
  ) {
    const float x0 = std::bit_cast<float>(words[offset + 0u]);
    const float y0 = std::bit_cast<float>(words[offset + 1u]);
    const float x1 = std::bit_cast<float>(words[offset + 2u]);
    const float y1 = std::bit_cast<float>(words[offset + 3u]);
    return std::isfinite(x0) && std::isfinite(y0) &&
           std::isfinite(x1) && std::isfinite(y1) &&
           x0 >= 0.0f && y0 >= 0.0f &&
           x1 <= 1.0f && y1 <= 1.0f &&
           x1 > x0 && y1 > y0;
  }

  [[nodiscard]] inline bool rect_contains(
    const roi_transform_words_t &words,
    const std::size_t outer_offset,
    const std::size_t inner_offset
  ) {
    const float ox0 = std::bit_cast<float>(words[outer_offset + 0u]);
    const float oy0 = std::bit_cast<float>(words[outer_offset + 1u]);
    const float ox1 = std::bit_cast<float>(words[outer_offset + 2u]);
    const float oy1 = std::bit_cast<float>(words[outer_offset + 3u]);
    const float ix0 = std::bit_cast<float>(words[inner_offset + 0u]);
    const float iy0 = std::bit_cast<float>(words[inner_offset + 1u]);
    const float ix1 = std::bit_cast<float>(words[inner_offset + 2u]);
    const float iy1 = std::bit_cast<float>(words[inner_offset + 3u]);
    return ix0 >= ox0 && iy0 >= oy0 && ix1 <= ox1 && iy1 <= oy1;
  }

  [[nodiscard]] inline bool physical_aspect_matches(
    const roi_transform_words_t &words,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height,
    const float relative_tolerance
  ) {
    const float x0 = std::bit_cast<float>(words[16u]);
    const float y0 = std::bit_cast<float>(words[17u]);
    const float x1 = std::bit_cast<float>(words[18u]);
    const float y1 = std::bit_cast<float>(words[19u]);
    const double crop_width =
      static_cast<double>(x1 - x0) * source_width;
    const double crop_height =
      static_cast<double>(y1 - y0) * source_height;
    const double left = crop_width * model_height;
    const double right = crop_height * model_width;
    const double scale = std::max({std::abs(left), std::abs(right), 1.0});
    return std::isfinite(left) && std::isfinite(right) &&
           std::abs(left - right) <= scale * relative_tolerance;
  }

  [[nodiscard]] inline std::uint32_t accepted_axis_start(
    const float value,
    const std::uint32_t extent
  ) {
    const double rounded = std::ceil(
      static_cast<double>(value) * extent - 0.5
    );
    return static_cast<std::uint32_t>(
      std::clamp(rounded, 0.0, static_cast<double>(extent))
    );
  }

  [[nodiscard]] inline bool valid_roi_transform(
    const roi_transform_words_t &words,
    const std::uint64_t expected_source_frame_id,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height
  ) {
    constexpr std::uint32_t valid_flag = 1u << 0u;
    constexpr std::uint32_t full_frame_flag = 1u << 1u;
    constexpr std::uint32_t active_roi_flag = 1u << 2u;
    constexpr std::uint32_t reset_debt_flag = 1u << 3u;
    constexpr std::uint32_t known_flags =
      valid_flag | full_frame_flag | active_roi_flag | reset_debt_flag;

    const auto flags = words[1u];
    const bool full_frame = (flags & full_frame_flag) != 0u;
    const bool active_roi = (flags & active_roi_flag) != 0u;
    const auto transform_version = join_u64(words[28u], words[29u]);
    const auto source_frame_id = join_u64(words[2u], words[3u]);
    if (
      words[0u] != models::frame_roi_transform_contract_version ||
      (flags & valid_flag) == 0u ||
      (flags & ~known_flags) != 0u ||
      full_frame == active_roi ||
      source_frame_id != expected_source_frame_id ||
      words[6u] != source_width ||
      words[7u] != source_height ||
      words[8u] != model_width ||
      words[9u] != model_height ||
      model_width == 0u || model_height == 0u ||
      model_width > max_depth_dimension ||
      model_height > max_depth_dimension ||
      model_width > source_width ||
      model_height > source_height ||
      model_width % models::frame_roi_model_patch_size != 0u ||
      model_height % models::frame_roi_model_patch_size != 0u ||
      transform_version == 0u ||
      words[30u] >= models::frame_roi_transform_bank_count ||
      !finite_rect(words, 12u) ||
      !finite_rect(words, 16u) ||
      !rect_contains(words, 16u, 12u)
    ) {
      return false;
    }

    for (std::size_t index = 24u; index < 28u; ++index) {
      const auto feather = std::bit_cast<float>(words[index]);
      if (!std::isfinite(feather) || feather < 0.0f) {
        return false;
      }
    }

    const auto x0 = accepted_axis_start(
      (std::bit_cast<float>(words[12u]) -
       std::bit_cast<float>(words[16u])) /
        (std::bit_cast<float>(words[18u]) -
         std::bit_cast<float>(words[16u])),
      model_width
    );
    const auto y0 = accepted_axis_start(
      (std::bit_cast<float>(words[13u]) -
       std::bit_cast<float>(words[17u])) /
        (std::bit_cast<float>(words[19u]) -
         std::bit_cast<float>(words[17u])),
      model_height
    );
    const auto x1 = accepted_axis_start(
      (std::bit_cast<float>(words[14u]) -
       std::bit_cast<float>(words[16u])) /
        (std::bit_cast<float>(words[18u]) -
         std::bit_cast<float>(words[16u])),
      model_width
    );
    const auto y1 = accepted_axis_start(
      (std::bit_cast<float>(words[15u]) -
       std::bit_cast<float>(words[17u])) /
        (std::bit_cast<float>(words[19u]) -
         std::bit_cast<float>(words[17u])),
      model_height
    );
    const std::uint64_t accepted_count =
      static_cast<std::uint64_t>(x1 - x0) * (y1 - y0);
    if (
      x1 <= x0 || y1 <= y0 ||
      words[20u] != x0 || words[21u] != y0 ||
      words[22u] != x1 || words[23u] != y1 ||
      words[10u] != accepted_count
    ) {
      return false;
    }

    if (full_frame) {
      constexpr std::array<std::uint32_t, 4> full_rect {
        0u,
        0u,
        0x3f800000u,
        0x3f800000u,
      };
      return std::equal(
               full_rect.begin(),
               full_rect.end(),
               words.begin() + 12u
             ) &&
             std::equal(
               full_rect.begin(),
               full_rect.end(),
               words.begin() + 16u
             ) &&
             words[20u] == 0u && words[21u] == 0u &&
             words[22u] == model_width && words[23u] == model_height &&
             words[10u] ==
               static_cast<std::uint64_t>(model_width) * model_height &&
             std::all_of(
               words.begin() + 24u,
               words.begin() + 28u,
               [](const auto word) {
                 return std::bit_cast<float>(word) == 0.0f;
               }
             ) &&
             physical_aspect_matches(
               words,
               source_width,
               source_height,
               model_width,
               model_height,
               0.02f
             );
    }

    const float focus_width =
      std::bit_cast<float>(words[14u]) -
      std::bit_cast<float>(words[12u]);
    const float focus_height =
      std::bit_cast<float>(words[15u]) -
      std::bit_cast<float>(words[13u]);
    return words[4u] != 0u &&
           words[5u] != 0u &&
           words[11u] != 0u &&
           std::bit_cast<float>(words[24u]) <= 0.25f * focus_width &&
           std::bit_cast<float>(words[26u]) <= 0.25f * focus_width &&
           std::bit_cast<float>(words[25u]) <= 0.25f * focus_height &&
           std::bit_cast<float>(words[27u]) <= 0.25f * focus_height &&
           physical_aspect_matches(
             words,
             source_width,
             source_height,
             model_width,
             model_height,
             0.0001f
           );
  }

  [[nodiscard]] inline bool valid_frame_metadata(
    const frame_metadata_t &value,
    const std::uint64_t expected_sequence,
    const std::uint32_t expected_source_width,
    const std::uint32_t expected_source_height,
    const std::uint32_t depth_reuse_interval,
    const bool allow_forced_current = false
  ) {
    const auto width = value.depth[0u];
    const auto height = value.depth[1u];
    const std::uint64_t pixels =
      static_cast<std::uint64_t>(width) * height;
    if (
      value.header != std::array<std::uint32_t, 4> {
        frame_metadata_magic,
        frame_metadata_schema,
        frame_metadata_word_count,
        roi_transform_word_offset,
      } ||
      width == 0u || height == 0u ||
      width > max_depth_dimension ||
      height > max_depth_dimension ||
      width > expected_source_width ||
      height > expected_source_height ||
      width % models::frame_roi_model_patch_size != 0u ||
      height % models::frame_roi_model_patch_size != 0u ||
      pixels > std::numeric_limits<std::uint32_t>::max() ||
      value.depth[2u] != pixels ||
      value.depth[3u] != sizeof(float) ||
      cache_sequence(value) != expected_sequence ||
      expected_sequence == 0u ||
      depth_reuse_interval == 0u ||
      depth_reuse_interval > 8u ||
      value.identity[0u] != expected_source_width ||
      value.identity[1u] != expected_source_height ||
      value.identity[2u] != width ||
      value.identity[3u] != height
    ) {
      return false;
    }

    const auto retained = retained_source_frame_id(value);
    if (transform_is_unbound_zero(value.roi_transform)) {
      if (retained != unbound_source_frame_id) {
        return false;
      }
      const double left =
        static_cast<double>(expected_source_width) * height;
      const double right =
        static_cast<double>(expected_source_height) * width;
      const double scale =
        std::max({std::abs(left), std::abs(right), 1.0});
      return std::abs(left - right) <= scale * 0.02;
    }
    const auto expected_retained =
      ((expected_sequence - 1u) / depth_reuse_interval) *
      depth_reuse_interval;
    const auto current_source_frame = expected_sequence - 1u;
    const bool forced_current =
      allow_forced_current && retained == current_source_frame;
    return
           (
             forced_current ||
             (
               retained <= expected_retained &&
               retained % depth_reuse_interval == 0u
             )
           ) &&
           valid_roi_transform(
             value.roi_transform,
             retained,
             expected_source_width,
             expected_source_height,
             width,
             height
           );
  }

  [[nodiscard]] inline bool valid_cached_state(
    const cached_state_words_t &words
  ) {
    for (const auto word : words) {
      if (!std::isfinite(std::bit_cast<float>(word))) {
        return false;
      }
    }
    const auto subject_initialized = std::bit_cast<float>(
      words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::initialized
      )]
    );
    const auto zero_anchor_valid = std::bit_cast<float>(
      words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::zero_anchor_valid
      )]
    );
    const auto cut_flags = std::bit_cast<float>(
      words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::cut_flags
      )]
    );
    const auto history_state = std::bit_cast<float>(
      words[sbs_adaptive_state::index(
        sbs_adaptive_state::word_e::model_input_history_state
      )]
    );
    const auto initialized = std::bit_cast<float>(
      words[sbs_adaptive_state::render_prefix_word_count + 2u]
    );
    const auto frame_state = std::bit_cast<float>(
      words[sbs_adaptive_state::render_prefix_word_count + 3u]
    );
    const bool frame_state_valid =
      frame_state == 0.0f ||
      (
        initialized == 1.0f &&
        (frame_state == 1.0f || frame_state == 2.0f)
      );
    return
      (subject_initialized == 0.0f || subject_initialized == 1.0f) &&
      (zero_anchor_valid == 0.0f || zero_anchor_valid == 1.0f) &&
      cut_flags >= 0.0f &&
      cut_flags <=
        static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) &&
      std::trunc(cut_flags) == cut_flags &&
      history_state >= 0.0f && history_state <= 4.0f &&
      std::trunc(history_state) == history_state &&
      (initialized == 0.0f || initialized == 1.0f) &&
      frame_state_valid;
  }

  [[nodiscard]] inline bool cached_state_requires_previous(
    const cached_state_words_t &state
  ) {
    const auto depth_state_offset =
      sbs_adaptive_state::render_prefix_word_count;
    const auto initialized =
      std::bit_cast<float>(state[depth_state_offset + 2u]);
    const auto frame_state =
      std::bit_cast<float>(state[depth_state_offset + 3u]);
    return initialized >= 0.5f && frame_state < 0.5f;
  }

  /**
   * Couple retained depth ownership to the cached DepthFrameState.
   *
   * A valid completion owns the exact cadence anchor. An invalid completion deliberately
   * preserves the prior packed frame and may retain only an earlier cadence-aligned
   * depth/transform pair. Every replay consumer uses this combined check so an old crop can
   * never masquerade as a current valid surface.
   */
  [[nodiscard]] inline bool valid_frame_metadata_for_state(
    const frame_metadata_t &metadata,
    const cached_state_words_t &state,
    const std::uint64_t expected_sequence,
    const std::uint32_t expected_source_width,
    const std::uint32_t expected_source_height,
    const std::uint32_t depth_reuse_interval,
    const bool allow_forced_current = false
  ) {
    if (
      !valid_frame_metadata(
        metadata,
        expected_sequence,
        expected_source_width,
        expected_source_height,
        depth_reuse_interval,
        allow_forced_current
      ) ||
      !valid_cached_state(state)
    ) {
      return false;
    }
    if (transform_is_unbound_zero(metadata.roi_transform)) {
      return true;
    }
    const auto cadence_retained =
      ((expected_sequence - 1u) / depth_reuse_interval) *
      depth_reuse_interval;
    const auto retained = retained_source_frame_id(metadata);
    const auto current_source_frame = expected_sequence - 1u;
    const bool forced_current =
      allow_forced_current && retained == current_source_frame;
    const auto expected_retained =
      forced_current ?
        current_source_frame :
        cadence_retained;
    const bool requires_previous =
      cached_state_requires_previous(state);
    return requires_previous ?
             retained < expected_retained :
             retained == expected_retained;
  }

  /**
   * A replay process starts with an empty packed target. Its first cache record
   * must therefore be self-contained even though later records may deliberately
   * preserve the preceding packed frame after an invalid depth completion.
   */
  [[nodiscard]] inline bool valid_scene_replay_frame(
    const frame_metadata_t &metadata,
    const cached_state_words_t &state,
    const std::uint64_t expected_sequence,
    const std::uint64_t scene_start_sequence,
    const std::uint32_t expected_source_width,
    const std::uint32_t expected_source_height,
    const std::uint32_t depth_reuse_interval,
    const bool allow_forced_current = false
  ) {
    return
      scene_start_sequence != 0u &&
      scene_start_sequence <= expected_sequence &&
      valid_frame_metadata_for_state(
        metadata,
        state,
        expected_sequence,
        expected_source_width,
        expected_source_height,
        depth_reuse_interval,
        allow_forced_current
      ) &&
      (
        expected_sequence != scene_start_sequence ||
        !cached_state_requires_previous(state)
      );
  }

  [[nodiscard]] constexpr std::uint64_t depth_payload_bytes(
    const frame_metadata_t &value
  ) {
    return static_cast<std::uint64_t>(value.depth[2u]) * sizeof(float);
  }
}  // namespace sbs_scene_cache
