/**
 * @file src/sbs_roi_shape_request.h
 * @brief Exact GPU/CPU ABI for one deterministic Host SBS ROI model-shape request.
 *
 * The request is deliberately small enough for one four-element StructuredBuffer<uint4>. The
 * controller state remains GPU-resident: CPU code may validate, compare, and carry this record,
 * but a 32-bit request id never substitutes for exact tuple comparison.
 */
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace models {
  inline constexpr std::uint32_t sbs_roi_shape_request_schema_version = 1u;
  inline constexpr std::uint32_t sbs_roi_shape_request_vector_count = 4u;
  inline constexpr std::uint32_t sbs_roi_shape_request_patch_size = 14u;
  inline constexpr std::uint32_t sbs_roi_shape_request_engine_max_dimension =
    1036u;
  inline constexpr float sbs_roi_shape_request_max_aspect_limit = 8.0f;
  inline constexpr float sbs_roi_full_frame_aspect_relative_tolerance =
    0.02f;
  // Shape-request ABI v1 is bound to scene-controller schema 1.0 encoded as float bits.
  inline constexpr std::uint32_t sbs_roi_shape_controller_schema_float_bits =
    0x3F800000u;

  enum class sbs_roi_shape_request_flag : std::uint32_t {
    valid = 1u << 0u,
    full_frame = 1u << 1u,
    active_roi = 1u << 2u,
    fallback = 1u << 3u,
    aspect_clamped = 1u << 4u,
    profile_clamped = 1u << 5u,
  };

  inline constexpr std::uint32_t sbs_roi_shape_request_known_flags =
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::valid) |
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::full_frame) |
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::active_roi) |
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::fallback) |
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::aspect_clamped) |
    static_cast<std::uint32_t>(sbs_roi_shape_request_flag::profile_clamped);

  enum class sbs_roi_shape_request_reason : std::uint32_t {
    none = 0u,
    inactive = 1u,
    invalid_source_or_profile = 2u,
    controller_schema_mismatch = 3u,
    backend_generation_mismatch = 4u,
    controller_output_invalid = 5u,
    controller_uninitialized = 6u,
    roi_not_locked = 7u,
    controller_fallback_active = 8u,
    invalid_rule_identity = 9u,
    malformed_committed_roi = 10u,
  };

  /**
   * Four-vector, 64-byte ABI. Float rectangle values are stored as their exact IEEE-754 bits.
   *
   * [0] header:   request schema, flags, reason, deterministic request id
   * [1] identity: backend generation, ROI generation, rule update count, rule-schema float bits
   * [2] shape:    source width, source height, model width, model height
   * [3] ROI:      committed source-UV x0, y0, x1, y1 float bits
   */
  struct alignas(16) sbs_roi_shape_request {
    std::array<std::uint32_t, 4> header {};
    std::array<std::uint32_t, 4> identity {};
    std::array<std::uint32_t, 4> shape {};
    std::array<std::uint32_t, 4> committed_roi_bits {};
  };

  static_assert(sizeof(sbs_roi_shape_request) == 64u);
  static_assert(alignof(sbs_roi_shape_request) == 16u);
  static_assert(std::is_standard_layout_v<sbs_roi_shape_request>);
  static_assert(std::is_trivially_copyable_v<sbs_roi_shape_request>);
  static_assert(offsetof(sbs_roi_shape_request, header) == 0u);
  static_assert(offsetof(sbs_roi_shape_request, identity) == 16u);
  static_assert(offsetof(sbs_roi_shape_request, shape) == 32u);
  static_assert(
    offsetof(sbs_roi_shape_request, committed_roi_bits) == 48u
  );

  [[nodiscard]] constexpr std::uint32_t sbs_roi_shape_flag_bits(
    sbs_roi_shape_request_flag flag
  ) {
    return static_cast<std::uint32_t>(flag);
  }

  [[nodiscard]] constexpr bool sbs_roi_shape_has_flag(
    const sbs_roi_shape_request &request,
    sbs_roi_shape_request_flag flag
  ) {
    return (request.header[1] & sbs_roi_shape_flag_bits(flag)) != 0u;
  }

  [[nodiscard]] constexpr std::uint32_t sbs_roi_shape_hash_word(
    std::uint32_t hash,
    std::uint32_t word
  ) {
    return (hash ^ word) * 16777619u;
  }

  /**
   * Diagnostic id shared with sbs_roi_shape_request.hlsl.
   *
   * Collisions are possible by construction. Consumers must compare the exact identity, shape,
   * and committed-ROI words before authorizing a transform or pairing a depth result.
   */
  [[nodiscard]] constexpr std::uint32_t sbs_roi_shape_request_id_from_fields(
    std::uint32_t request_schema,
    std::uint32_t backend_generation,
    std::uint32_t roi_generation,
    std::uint32_t sampled_rule_update_count,
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t model_width,
    std::uint32_t model_height,
    const std::array<std::uint32_t, 4> &committed_roi_bits
  ) {
    std::uint32_t hash = 2166136261u;
    hash = sbs_roi_shape_hash_word(hash, request_schema);
    hash = sbs_roi_shape_hash_word(hash, backend_generation);
    hash = sbs_roi_shape_hash_word(hash, roi_generation);
    hash = sbs_roi_shape_hash_word(hash, sampled_rule_update_count);
    hash = sbs_roi_shape_hash_word(hash, source_width);
    hash = sbs_roi_shape_hash_word(hash, source_height);
    hash = sbs_roi_shape_hash_word(hash, model_width);
    hash = sbs_roi_shape_hash_word(hash, model_height);
    for (const auto word : committed_roi_bits) {
      hash = sbs_roi_shape_hash_word(hash, word);
    }
    return hash == 0u ? 0xA511E9B3u : hash;
  }

  [[nodiscard]] constexpr std::uint32_t sbs_roi_shape_request_id(
    const sbs_roi_shape_request &request
  ) {
    return sbs_roi_shape_request_id_from_fields(
      request.header[0],
      request.identity[0],
      request.identity[1],
      request.identity[2],
      request.shape[0],
      request.shape[1],
      request.shape[2],
      request.shape[3],
      request.committed_roi_bits
    );
  }

  // CPU-side ABI vectors. These exact values make field-order or hash changes fail compilation.
  static_assert(
    sbs_roi_shape_request_id_from_fields(
      1u,
      7u,
      3u,
      42u,
      3840u,
      2160u,
      770u,
      434u,
      std::array<std::uint32_t, 4> {
        0x3E000000u,
        0x3E800000u,
        0x3F600000u,
        0x3F400000u,
      }
    ) == 0x7C7D2C40u
  );
  static_assert(
    sbs_roi_shape_request_id_from_fields(
      1u,
      7u,
      3u,
      43u,
      3840u,
      2160u,
      770u,
      434u,
      std::array<std::uint32_t, 4> {
        0x3E000000u,
        0x3E800000u,
        0x3F600000u,
        0x3F400000u,
      }
    ) == 0xF7FF276Fu
  );

  [[nodiscard]] constexpr bool sbs_roi_shape_request_id_matches(
    const sbs_roi_shape_request &request
  ) {
    return request.header[3] != 0u &&
           request.header[3] == sbs_roi_shape_request_id(request);
  }

  [[nodiscard]] inline bool sbs_roi_shape_rect_valid(
    const std::array<std::uint32_t, 4> &bits
  ) {
    const float x0 = std::bit_cast<float>(bits[0]);
    const float y0 = std::bit_cast<float>(bits[1]);
    const float x1 = std::bit_cast<float>(bits[2]);
    const float y1 = std::bit_cast<float>(bits[3]);
    return std::isfinite(x0) && std::isfinite(y0) &&
           std::isfinite(x1) && std::isfinite(y1) &&
           x0 >= 0.0f && y0 >= 0.0f && x1 <= 1.0f && y1 <= 1.0f &&
           x1 > x0 && y1 > y0;
  }

  [[nodiscard]] constexpr bool sbs_roi_shape_patch_aligned(
    std::uint32_t value
  ) {
    return value >= sbs_roi_shape_request_patch_size &&
           value % sbs_roi_shape_request_patch_size == 0u;
  }

  /**
   * Preserve the source aspect while applying `max_budget_aspect` as an inference-area budget.
   *
   * Clamping the tensor's aspect itself stretches the full source into a different geometry.
   * Instead, reduce the requested short side by sqrt(budget/actual) whenever a landscape or
   * portrait source exceeds the configured aspect envelope. The resulting tensor retains the
   * source aspect and uses no more pixels than the old capped-aspect tensor.
   */
  [[nodiscard]] inline float sbs_roi_budgeted_full_frame_short_side(
    const float source_aspect,
    const float requested_short_side,
    const float max_budget_aspect
  ) {
    if (!std::isfinite(source_aspect) ||
        !std::isfinite(requested_short_side) ||
        !std::isfinite(max_budget_aspect) ||
        source_aspect <= 0.0f ||
        requested_short_side <= 0.0f ||
        max_budget_aspect < 1.0f) {
      return 0.0f;
    }
    const float physical_aspect =
      source_aspect >= 1.0f ? source_aspect : 1.0f / source_aspect;
    const float budget_aspect =
      physical_aspect < max_budget_aspect ?
        physical_aspect :
        max_budget_aspect;
    return requested_short_side *
           std::sqrt(budget_aspect / physical_aspect);
  }

  /**
   * Verify that a canonical full-frame tensor preserves the source's physical geometry.
   *
   * Full-frame tensors may legitimately exceed the active-ROI aspect cap. They must instead
   * match the source aspect closely enough that the full-frame UV transform cannot stretch it.
   * Cross-products avoid division and preserve the exact integer raster relationship.
   */
  [[nodiscard]] inline bool sbs_roi_full_frame_shape_matches(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const std::uint32_t model_width,
    const std::uint32_t model_height,
    const float relative_tolerance =
      sbs_roi_full_frame_aspect_relative_tolerance
  ) {
    if (source_width == 0u ||
        source_height == 0u ||
        model_width == 0u ||
        model_height == 0u ||
        !std::isfinite(relative_tolerance) ||
        relative_tolerance < 0.0f) {
      return false;
    }

    const double source_cross =
      static_cast<double>(source_width) *
      static_cast<double>(model_height);
    const double model_cross =
      static_cast<double>(source_height) *
      static_cast<double>(model_width);
    const double scale = std::max(
      std::max(std::abs(source_cross), std::abs(model_cross)),
      1.0
    );
    return std::abs(source_cross - model_cross) <=
           scale * static_cast<double>(relative_tolerance);
  }

  /**
   * Select a patch-aligned full-frame tensor without changing the source's physical aspect.
   *
   * The configured aspect is an area budget (via the helper above), while native/profile maxima
   * can reduce the short side further. Independent axis rounding is deliberately avoided because
   * it accumulates enough aspect error to make a nominal full-frame transform unsafe.
   */
  [[nodiscard]] inline std::array<std::uint32_t, 2>
  sbs_roi_full_frame_model_shape(
    float source_aspect,
    float requested_short_side,
    float max_budget_aspect,
    std::uint32_t max_width,
    std::uint32_t max_height,
    std::uint32_t patch = sbs_roi_shape_request_patch_size
  ) {
    if (!std::isfinite(source_aspect) ||
        source_aspect <= 0.0f ||
        patch == 0u ||
        max_width < patch ||
        max_height < patch) {
      return {};
    }
    const float budgeted_short_side =
      sbs_roi_budgeted_full_frame_short_side(
        source_aspect,
        requested_short_side,
        max_budget_aspect
      );
    if (!(budgeted_short_side > 0.0f)) {
      return {};
    }

    const auto align_nearest = [patch](const float value) {
      if (!std::isfinite(value) || value <= 0.0f) {
        return 0u;
      }
      const double patch_units = std::round(
        static_cast<double>(value) / static_cast<double>(patch)
      );
      const double maximum_patch_units = static_cast<double>(
        std::numeric_limits<std::uint32_t>::max() / patch
      );
      if (!std::isfinite(patch_units) ||
          patch_units > maximum_patch_units) {
        return 0u;
      }
      return std::max(
        patch,
        static_cast<std::uint32_t>(patch_units) * patch
      );
    };
    max_width = std::max(
      patch,
      (max_width / patch) * patch
    );
    max_height = std::max(
      patch,
      (max_height / patch) * patch
    );
    const auto aspect_matches = [source_aspect](
                                  const std::uint32_t width,
                                  const std::uint32_t height) {
      const float expected_width =
        source_aspect * static_cast<float>(height);
      const float actual_width = static_cast<float>(width);
      const float scale = std::max(
        std::max(std::abs(expected_width), std::abs(actual_width)),
        1.0f
      );
      return std::isfinite(expected_width) &&
             std::abs(expected_width - actual_width) <=
               scale * sbs_roi_full_frame_aspect_relative_tolerance;
    };
    const auto requested_short = align_nearest(budgeted_short_side);
    if (requested_short == 0u) {
      return {};
    }
    if (source_aspect >= 1.0f) {
      for (auto height = std::min(requested_short, max_height);
           height >= patch;
           height -= patch) {
        const auto width = align_nearest(
          static_cast<float>(height) * source_aspect
        );
        if (width <= max_width && aspect_matches(width, height)) {
          return {width, height};
        }
        if (height == patch) {
          break;
        }
      }
    } else {
      for (auto width = std::min(requested_short, max_width);
           width >= patch;
           width -= patch) {
        const auto height = align_nearest(
          static_cast<float>(width) / source_aspect
        );
        if (height <= max_height && aspect_matches(width, height)) {
          return {width, height};
        }
        if (width == patch) {
          break;
        }
      }
    }
    return {};
  }

  [[nodiscard]] constexpr bool sbs_roi_shape_reason_known(
    std::uint32_t reason
  ) {
    return reason <= static_cast<std::uint32_t>(
      sbs_roi_shape_request_reason::malformed_committed_roi
    );
  }

  /**
   * Validate ABI invariants that are self-contained in the request.
   *
   * Canonical fallback dimensions and session-specific profile caps are validated by the overload
   * below because they are external policy, not encoded redundantly in this 64-byte record.
   */
  [[nodiscard]] inline bool sbs_roi_shape_request_valid(
    const sbs_roi_shape_request &request
  ) {
    const auto flags = request.header[1];
    const bool full_frame =
      sbs_roi_shape_has_flag(request, sbs_roi_shape_request_flag::full_frame);
    const bool active_roi =
      sbs_roi_shape_has_flag(request, sbs_roi_shape_request_flag::active_roi);
    const bool fallback =
      sbs_roi_shape_has_flag(request, sbs_roi_shape_request_flag::fallback);
    const auto reason = request.header[2];
    const auto source_width = request.shape[0];
    const auto source_height = request.shape[1];
    const auto model_width = request.shape[2];
    const auto model_height = request.shape[3];

    if (request.header[0] != sbs_roi_shape_request_schema_version ||
        (flags & ~sbs_roi_shape_request_known_flags) != 0u ||
        !sbs_roi_shape_has_flag(request, sbs_roi_shape_request_flag::valid) ||
        full_frame == active_roi ||
        !sbs_roi_shape_reason_known(reason) ||
        fallback != (reason != 0u) ||
        source_width == 0u || source_height == 0u ||
        !sbs_roi_shape_patch_aligned(model_width) ||
        !sbs_roi_shape_patch_aligned(model_height) ||
        model_width > source_width || model_height > source_height ||
        model_width > sbs_roi_shape_request_engine_max_dimension ||
        model_height > sbs_roi_shape_request_engine_max_dimension ||
        !sbs_roi_shape_request_id_matches(request)) {
      return false;
    }

    if (active_roi) {
      return !fallback && reason == 0u &&
             request.identity[0] != 0u &&
             request.identity[1] != 0u &&
             request.identity[2] != 0u &&
             request.identity[3] ==
               sbs_roi_shape_controller_schema_float_bits &&
             sbs_roi_shape_rect_valid(request.committed_roi_bits);
    }

    return fallback && reason != 0u;
  }

  struct sbs_roi_shape_request_limits {
    std::uint32_t canonical_model_width = 0u;
    std::uint32_t canonical_model_height = 0u;
    std::uint32_t profile_max_width =
      sbs_roi_shape_request_engine_max_dimension;
    std::uint32_t profile_max_height =
      sbs_roi_shape_request_engine_max_dimension;
    float max_model_aspect = 4.0f;
  };

  [[nodiscard]] inline bool sbs_roi_shape_request_valid(
    const sbs_roi_shape_request &request,
    const sbs_roi_shape_request_limits &limits
  ) {
    if (!sbs_roi_shape_request_valid(request) ||
        !sbs_roi_shape_patch_aligned(limits.canonical_model_width) ||
        !sbs_roi_shape_patch_aligned(limits.canonical_model_height) ||
        limits.profile_max_width < sbs_roi_shape_request_patch_size ||
        limits.profile_max_height < sbs_roi_shape_request_patch_size ||
        !std::isfinite(limits.max_model_aspect) ||
        limits.max_model_aspect < 1.0f ||
        limits.max_model_aspect >
          sbs_roi_shape_request_max_aspect_limit ||
        request.shape[2] > limits.profile_max_width ||
        request.shape[3] > limits.profile_max_height) {
      return false;
    }

    if (sbs_roi_shape_has_flag(
          request,
          sbs_roi_shape_request_flag::full_frame
        )) {
      // The canonical tensor always preserves the complete source aspect. For a source wider or
      // taller than the active-ROI envelope, canonical sizing reduces its short side to keep the
      // same pixel budget instead of squeezing the image. The active cap therefore does not apply
      // to this exact canonical fallback.
      return request.shape[2] == limits.canonical_model_width &&
             request.shape[3] == limits.canonical_model_height &&
             sbs_roi_full_frame_shape_matches(
               request.shape[0],
               request.shape[1],
               request.shape[2],
               request.shape[3]
             );
    }

    const float aspect =
      static_cast<float>(request.shape[2]) /
      static_cast<float>(request.shape[3]);
    return aspect <= limits.max_model_aspect &&
           aspect >= 1.0f / limits.max_model_aspect;
  }

  /** Exact tuple carried into the transform-builder submission constants. */
  struct sbs_roi_shape_request_expectation {
    std::uint32_t request_schema = 0u;
    std::uint32_t backend_generation = 0u;
    std::uint32_t roi_generation = 0u;
    std::uint32_t rule_update_count = 0u;
    std::uint32_t shape_request_id = 0u;
    std::uint32_t source_width = 0u;
    std::uint32_t source_height = 0u;
    std::uint32_t model_width = 0u;
    std::uint32_t model_height = 0u;
    std::array<std::uint32_t, 4> committed_roi_bits {};
  };

  [[nodiscard]] constexpr sbs_roi_shape_request_expectation
  sbs_roi_shape_expectation(const sbs_roi_shape_request &request) {
    return {
      request.header[0],
      request.identity[0],
      request.identity[1],
      request.identity[2],
      request.header[3],
      request.shape[0],
      request.shape[1],
      request.shape[2],
      request.shape[3],
      request.committed_roi_bits,
    };
  }

  [[nodiscard]] constexpr bool sbs_roi_shape_matches_exact(
    const sbs_roi_shape_request &request,
    const sbs_roi_shape_request_expectation &expected
  ) {
    return request.header[0] == expected.request_schema &&
           request.identity[0] == expected.backend_generation &&
           request.identity[1] == expected.roi_generation &&
           request.identity[2] == expected.rule_update_count &&
           request.header[3] == expected.shape_request_id &&
           request.shape[0] == expected.source_width &&
           request.shape[1] == expected.source_height &&
           request.shape[2] == expected.model_width &&
           request.shape[3] == expected.model_height &&
           request.committed_roi_bits == expected.committed_roi_bits;
  }

  [[nodiscard]] constexpr bool sbs_roi_shape_same_bound_rule_state(
    const sbs_roi_shape_request &first,
    const sbs_roi_shape_request &second
  ) {
    return first.header[0] == second.header[0] &&
           first.identity[0] == second.identity[0] &&
           first.identity[1] == second.identity[1] &&
           first.shape[0] == second.shape[0] &&
           first.shape[1] == second.shape[1] &&
           first.committed_roi_bits == second.committed_roi_bits;
  }

  /**
   * Validate a delayed request against a newer controller snapshot without starving the async
   * handoff. The sampled update count remains provenance and part of the request id, while a
   * later count is accepted only for the same backend/ROI generation and exact committed ROI.
   */
  [[nodiscard]] constexpr bool sbs_roi_shape_current_rule_matches(
    const sbs_roi_shape_request &request,
    std::uint32_t current_backend_generation,
    std::uint32_t current_roi_generation,
    std::uint32_t current_rule_update_count,
    std::uint32_t current_source_width,
    std::uint32_t current_source_height,
    const std::array<std::uint32_t, 4> &current_committed_roi_bits
  ) {
    return request.identity[0] == current_backend_generation &&
           request.identity[1] == current_roi_generation &&
           current_rule_update_count >= request.identity[2] &&
           request.shape[0] == current_source_width &&
           request.shape[1] == current_source_height &&
           request.committed_roi_bits == current_committed_roi_bits;
  }

  enum class sbs_roi_shape_request_relation {
    invalid,
    identical,
    same_rule_different_policy,
    different_rule_state,
  };

  [[nodiscard]] inline sbs_roi_shape_request_relation
  sbs_roi_shape_classify(
    const sbs_roi_shape_request &previous,
    const sbs_roi_shape_request &current
  ) {
    if (!sbs_roi_shape_request_valid(current)) {
      return sbs_roi_shape_request_relation::invalid;
    }
    if (previous.header == current.header &&
        previous.identity == current.identity &&
        previous.shape == current.shape &&
        previous.committed_roi_bits == current.committed_roi_bits) {
      return sbs_roi_shape_request_relation::identical;
    }
    return sbs_roi_shape_same_bound_rule_state(previous, current) ?
      sbs_roi_shape_request_relation::same_rule_different_policy :
      sbs_roi_shape_request_relation::different_rule_state;
  }
}  // namespace models
