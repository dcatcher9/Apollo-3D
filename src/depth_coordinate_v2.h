/**
 * @file src/depth_coordinate_v2.h
 * @brief CPU-side contract for the production Host SBS raw-coordinate V2 GPU pipeline.
 */
#pragma once

#include <cstdint>

#include "generated/depth_coordinate_v2_contract.h"

namespace models::depth_coordinate_v2 {
  // File schemas are intentionally independent of the GPU/algorithm contract schema. A JSON
  // layout change must bump the relevant value without pretending the coordinate math changed.
  // Dump 3D retains the historical `shadow_*` filenames as a compatibility schema; these values
  // now describe production V2 evidence, not an alternate renderer.
  inline constexpr std::uint32_t shadow_state_dump_schema = 7u;
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
    const std::uint32_t calibration_revision_bits
  ) {
    std::uint32_t checksum = 0u;
    checksum = (checksum ^ center_bits) * 16777619u;
    checksum = (checksum ^ inverse_scale_bits) * 16777619u;
    checksum = (checksum ^ calibration_revision_bits) * 16777619u;
    return checksum;
  }

  constexpr bool camera_center_integrity_is_valid(
    const std::uint32_t center_bits,
    const std::uint32_t inverse_scale_bits,
    const std::uint32_t calibration_revision_bits,
    const std::uint32_t integrity_bits
  ) {
    return integrity_bits == camera_center_integrity_for_words(
      center_bits,
      inverse_scale_bits,
      calibration_revision_bits
    );
  }

  static_assert(camera_center_integrity_for_words(0u, 0u, 0u) == 0u);

  inline constexpr float parallax_gain = gain_per_pop * reference_pop_strength;

  constexpr float near_tail_dense_weight_for_coverage(const float coverage) {
    const float normalized =
      (coverage - near_tail_coverage_low) /
      (near_tail_coverage_high - near_tail_coverage_low);
    const float clamped = normalized < 0.0f ? 0.0f : normalized > 1.0f ? 1.0f : normalized;
    return clamped * clamped * (3.0f - 2.0f * clamped);
  }

  constexpr float near_tail_effective_tau_for_coverage(const float coverage) {
    return near_log_tau +
      (near_log_tau_dense - near_log_tau) * near_tail_dense_weight_for_coverage(coverage);
  }

  static_assert(near_tail_effective_tau_for_coverage(0.0f) == near_log_tau);
  static_assert(near_tail_effective_tau_for_coverage(1.0f) == near_log_tau_dense);

  // V2 has one artistic authority: the configured base pop. Legacy adaptive ceilings and state
  // are intentionally absent from this contract.
  constexpr float requested_pop_strength(const float configured_pop) {
    return configured_pop > 0.0f ? configured_pop : 0.0f;
  }

  constexpr float requested_gain_for_config(const float configured_pop) {
    return gain_per_pop * requested_pop_strength(configured_pop);
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
