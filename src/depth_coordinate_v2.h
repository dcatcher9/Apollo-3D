/**
 * @file src/depth_coordinate_v2.h
 * @brief CPU-side contract for the production Host SBS raw-coordinate V2 GPU pipeline.
 */
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#include "generated/depth_coordinate_v2_contract.h"

namespace models::depth_coordinate_v2 {
  inline constexpr std::uint32_t stage_histogram_source_bin_count = 256u;
  inline constexpr std::uint32_t stage_histogram_bin_count = 128u;

  constexpr bool convergence_curve_is_valid(const float value) {
    return value == convergence_curve_default;
  }

  // File schemas are intentionally independent of the GPU/algorithm contract schema. A JSON
  // layout change must bump the relevant value without pretending the coordinate math changed.
  // Dump 3D retains the historical `shadow_*` filenames as a compatibility schema; these values
  // now describe production V2 evidence, not an alternate renderer.
  inline constexpr std::uint32_t shadow_state_dump_schema = 15u;
  inline constexpr std::uint32_t shadow_frame_stats_dump_schema = 2u;
  inline constexpr std::uint32_t reserved_calibration_revision = 0xffffffffu;

  constexpr bool calibration_revision_is_valid(const std::uint32_t revision) {
    return revision != reserved_calibration_revision;
  }

  static_assert(calibration_revision_is_valid(0u));
  static_assert(calibration_revision_is_valid(1u));
  static_assert(!calibration_revision_is_valid(reserved_calibration_revision));

  constexpr std::uint32_t camera_center_integrity_for_words(
    const std::uint32_t center_bits,
    const std::uint32_t inverse_scale_bits,
    const std::uint32_t convergence_curve_bits,
    const std::uint32_t calibration_revision_bits
  ) {
    std::uint32_t checksum = 0u;
    checksum = (checksum ^ center_bits) * 16777619u;
    checksum = (checksum ^ inverse_scale_bits) * 16777619u;
    checksum = (checksum ^ convergence_curve_bits) * 16777619u;
    checksum = (checksum ^ calibration_revision_bits) * 16777619u;
    return checksum;
  }

  constexpr bool camera_center_integrity_is_valid(
    const std::uint32_t center_bits,
    const std::uint32_t inverse_scale_bits,
    const std::uint32_t convergence_curve_bits,
    const std::uint32_t calibration_revision_bits,
    const std::uint32_t integrity_bits
  ) {
    return integrity_bits == camera_center_integrity_for_words(
      center_bits,
      inverse_scale_bits,
      convergence_curve_bits,
      calibration_revision_bits
    );
  }

  static_assert(camera_center_integrity_for_words(0u, 0u, 0u, 0u) == 0u);

  /** Authenticate a serialized ParallaxState before exposing it to the compact live renderer. */
  inline bool parallax_state_words_are_authenticated(
    const state_words_t &words,
    const float raw_coordinate_scale
  ) {
    const auto scalar = [&words](const std::size_t index) {
      return std::bit_cast<float>(words[index]);
    };
    const float center_value = scalar(center);
    const float inverse_scale_value = scalar(inverse_scale);
    const float convergence_value = scalar(convergence_curve);
    const float container_value = scalar(container_scale);
    const float frame_valid_value = scalar(frame_valid);
    const auto revision = words[calibration_revision];
    const bool frame_is_valid = frame_valid_value == 1.0f;
    if (!std::isfinite(raw_coordinate_scale) || raw_coordinate_scale <= 0.0f ||
        words[contract_tag_bits] != contract_tag ||
        words[mapping_state_reserved_1] != 0u ||
        words[mapping_state_reserved_2] != 0u ||
        !std::isfinite(center_value) || !std::isfinite(inverse_scale_value) ||
        !std::isfinite(convergence_value) || !std::isfinite(container_value) ||
        (frame_valid_value != 0.0f && frame_valid_value != 1.0f) ||
        !convergence_curve_is_valid(convergence_value) || container_value != 1.0f ||
        words[renderer_authorization_bits] !=
          (frame_is_valid ? contract_tag : 0u) ||
        !camera_center_integrity_is_valid(
          words[center],
          words[inverse_scale],
          words[convergence_curve],
          revision,
          words[camera_center_integrity_bits]
        )) {
      return false;
    }

    const bool camera_initialized =
      inverse_scale_value > 0.0f && revision > 0u &&
      calibration_revision_is_valid(revision) &&
      std::abs(1.0f / inverse_scale_value - raw_coordinate_scale) <= 2.0e-6f;
    const bool camera_empty =
      center_value == 0.0f && inverse_scale_value == 0.0f &&
      (revision == 0u || calibration_revision_is_valid(revision));
    return frame_is_valid ? camera_initialized : (camera_initialized || camera_empty);
  }

  inline constexpr float parallax_gain = gain_per_pop * reference_pop_strength;

  // V2 has one artistic authority: the configured base pop. Legacy adaptive ceilings and state
  // are intentionally absent from this contract.
  constexpr float requested_pop_strength(const float configured_pop) {
    return configured_pop > 0.0f ? configured_pop : 0.0f;
  }

  constexpr float requested_gain_for_config(const float configured_pop) {
    return gain_per_pop * requested_pop_strength(configured_pop);
  }

  inline float pointwise_container(
    const float requested,
    const float limit = direct_container_limit
  ) {
    if (!std::isfinite(requested) || !std::isfinite(limit) || limit <= 0.0f) {
      return 0.0f;
    }
    const float requested_magnitude = std::abs(requested);
    const float smaller = std::min(requested_magnitude, limit);
    const float larger = std::max(requested_magnitude, limit);
    const float ratio = smaller / larger;
    const float ratio_squared = ratio * ratio;
    const float fourth_root = std::sqrt(std::sqrt(
      1.0f + ratio_squared * ratio_squared
    ));
    const float contained = std::copysign(smaller / fourth_root, requested);
    return std::clamp(contained, -limit, limit);
  }

  constexpr bool cut_generation_changed(
    const std::uint32_t previous_count,
    const std::uint32_t current_count,
    const bool pulse
  ) {
    return pulse || current_count != previous_count;
  }

  template<class CalibrationRange>
  constexpr const model_calibration_t *find_model_calibration_in(
    const CalibrationRange &calibrations,
    const std::string_view depth_model,
    const std::string_view depth_model_url,
    const std::string_view onnx_sha256
  ) {
    const model_calibration_t *match = nullptr;
    for (const auto &calibration : calibrations) {
      if (calibration.depth_model == depth_model &&
          calibration.depth_model_url == depth_model_url &&
          calibration.onnx_sha256 == onnx_sha256) {
        if (match) {
          return nullptr;
        }
        match = &calibration;
      }
    }
    return match;
  }

  constexpr const model_calibration_t *find_model_calibration(
    const std::string_view depth_model,
    const std::string_view depth_model_url,
    const std::string_view onnx_sha256
  ) {
    return find_model_calibration_in(
      model_calibrations,
      depth_model,
      depth_model_url,
      onnx_sha256
    );
  }

  constexpr const model_calibration_t *find_capture_calibration(
    const std::string_view depth_model,
    const std::string_view depth_model_url,
    const std::string_view onnx_sha256,
    const std::string_view preprocess_profile,
    const std::string_view preprocess_source_closure_sha256,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    const auto *calibration = find_model_calibration(
      depth_model,
      depth_model_url,
      onnx_sha256
    );
    return calibration && calibration->preprocess.profile == preprocess_profile &&
             calibration->preprocess.source_closure_sha256 ==
               preprocess_source_closure_sha256 &&
             model_calibration_supports_shape(*calibration, width, height) ?
             calibration :
             nullptr;
  }

  constexpr bool capture_identity_is_calibrated(
    const std::string_view depth_model,
    const std::string_view depth_model_url,
    const std::string_view onnx_sha256,
    const std::string_view preprocess_profile,
    const std::string_view preprocess_source_closure_sha256,
    const std::uint32_t width,
    const std::uint32_t height
  ) {
    return find_capture_calibration(
             depth_model,
             depth_model_url,
             onnx_sha256,
             preprocess_profile,
             preprocess_source_closure_sha256,
             width,
             height
           ) != nullptr;
  }
}  // namespace models::depth_coordinate_v2
