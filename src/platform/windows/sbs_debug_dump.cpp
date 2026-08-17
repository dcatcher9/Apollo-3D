/**
 * @file src/platform/windows/sbs_debug_dump.cpp
 * @brief Transactional, same-frame Host-SBS diagnostic package writer.
 */
#include "sbs_debug_dump.h"

// standard includes
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// platform includes
#include <windows.h>
#include <wrl/client.h>

// lib includes
#include <nlohmann/json.hpp>
#include <zlib.h>

// local includes
#include "src/generated/sbs_adaptive_state_contract.h"
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_resolution.h"
#include "src/host_sbs_gpu_trace.h"
#include "src/host_sbs_shader_cache.h"
#include "src/logging.h"
#include "src/model_manager.h"
#include "src/video_depth_estimator.h"
#include "sbs_debug_dump_async.h"

namespace platf::sbs_debug {

  using namespace std::literals;

  namespace {

    using namespace models::depth_coordinate_v2;

    static_assert(
      std::endian::native == std::endian::little,
      "Dump contracts explicitly use little-endian float and integer words."
    );

    constexpr auto retry_backoff = std::chrono::seconds(1);

    bool parallax_v2_shader_identity_matches_contract(
      const std::shared_ptr<const models::parallax_v2_shader_provenance_t> &identity
    ) {
      return identity &&
             models::parallax_v2_shader_provenance_matches_current_contract(*identity);
    }

    nlohmann::json parallax_v2_shader_identity_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      return {
        {"source_closure_schema", identity.source_closure_schema},
        {"source_compile_flags", identity.source_compile_flags},
        {"source_macro_count", identity.source_macro_count},
        {"source_closure_sha256", identity.source_closure_sha256},
      };
    }

    bool gpu_trace_shader_identity_matches_contract(
      const std::shared_ptr<const models::host_sbs_gpu_trace_provenance_t> &identity
    ) {
      return identity &&
             identity->source_closure_schema ==
               models::host_sbs_shader_cache::source_closure_schema &&
             identity->source_compile_flags ==
               models::host_sbs_shader_cache::shader_compile_flags &&
             identity->source_macro_count == 0u &&
             identity->source_closure_sha256 ==
               models::host_sbs_shader_cache::gpu_trace_source_closure_sha256;
    }

    nlohmann::json parallax_v2_coordinate_binding(
      const models::parallax_v2_shader_provenance_t &identity,
      const char *word_count_key,
      const std::size_t word_count
    ) {
      using namespace models::depth_coordinate_v2;
      auto binding = parallax_v2_shader_identity_json(identity);
      binding["schema"] = contract_schema;
      binding["tag"] = contract_tag;
      binding[word_count_key] = word_count;
      return binding;
    }

    constexpr std::uint32_t subtitle_locator_flag_owner = 1u;
    constexpr std::uint32_t subtitle_locator_flag_pending = 2u;
    constexpr std::uint32_t subtitle_locator_flag_target_valid = 4u;
    constexpr std::uint32_t subtitle_locator_flag_target_reset = 8u;
    constexpr std::uint32_t subtitle_locator_flag_provisional_current =
      subtitle_locator_provisional_current_flag;
    constexpr std::uint32_t subtitle_locator_known_flags =
      subtitle_locator_flag_owner |
      subtitle_locator_flag_pending |
      subtitle_locator_flag_target_valid |
      subtitle_locator_flag_target_reset |
      subtitle_locator_flag_provisional_current;
    constexpr std::uint32_t subtitle_locator_max_event = 3u;
    constexpr std::uint32_t subtitle_locator_max_fade = 2u;

    nlohmann::json subtitle_shader_contract_json(
      const models::parallax_v2_shader_provenance_t &identity,
      nlohmann::json source_specs
    ) {
      using namespace models::depth_coordinate_v2;
      auto contract = parallax_v2_shader_identity_json(identity);
      contract["depth_coordinate_v2_schema"] = contract_schema;
      contract["depth_coordinate_v2_tag"] = contract_tag;
      contract["source_specs"] = std::move(source_specs);
      return contract;
    }

    nlohmann::json subtitle_ocr_producer_contract_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      using namespace models::depth_coordinate_v2;
      return {
        {"contract_schema", subtitle_ocr_contract_schema},
        {"record_schema", subtitle_ocr_record_schema},
        {"record_tag", subtitle_ocr_record_tag},
        {"record_word_count", subtitle_ocr_record_word_count},
        {"raw_box_capacity", subtitle_ocr_raw_box_capacity},
        {"final_box_capacity", subtitle_ocr_final_box_capacity},
        {"model", {
          {"name", std::string {subtitle_ocr_model_name}},
          {"asset_path", std::string {subtitle_ocr_asset_path}},
          {"artifact_onnx_sha256", std::string {subtitle_ocr_artifact_onnx_sha256}},
          {"source_url", std::string {subtitle_ocr_source_url}},
          {"source_onnx_sha256", std::string {subtitle_ocr_source_onnx_sha256}},
          {"conversion_tool", std::string {subtitle_ocr_conversion_tool}},
          {"conversion_version", std::string {subtitle_ocr_conversion_version}},
          {"conversion_recipe", std::string {subtitle_ocr_conversion_recipe}},
          {"conversion_calibration_profile", std::string {
            subtitle_ocr_conversion_calibration_profile
          }},
          {"engine_recipe", std::string {subtitle_ocr_engine_recipe}},
          {"preprocess_profile", std::string {subtitle_ocr_preprocess_profile}},
          {"source_crop", std::string {subtitle_ocr_source_crop}},
          {"input", {
            {"name", std::string {subtitle_ocr_input_name}},
            {"dtype", std::string {subtitle_ocr_input_dtype}},
            {"layout", std::string {subtitle_ocr_input_layout}},
            {"shape", {
              subtitle_ocr_input_n,
              subtitle_ocr_input_c,
              subtitle_ocr_input_height,
              subtitle_ocr_input_width,
            }},
            {"channels", {
              std::string {subtitle_ocr_input_channels[0]},
              std::string {subtitle_ocr_input_channels[1]},
              std::string {subtitle_ocr_input_channels[2]},
            }},
            {"imagenet_mean", {
              subtitle_ocr_imagenet_mean[0],
              subtitle_ocr_imagenet_mean[1],
              subtitle_ocr_imagenet_mean[2],
            }},
            {"imagenet_std", {
              subtitle_ocr_imagenet_std[0],
              subtitle_ocr_imagenet_std[1],
              subtitle_ocr_imagenet_std[2],
            }},
          }},
          {"output", {
            {"name", std::string {subtitle_ocr_output_name}},
            {"dtype", std::string {subtitle_ocr_output_dtype}},
            {"layout", std::string {subtitle_ocr_output_layout}},
            {"shape", {
              subtitle_ocr_output_n,
              subtitle_ocr_output_c,
              subtitle_ocr_output_height,
              subtitle_ocr_output_width,
            }},
          }},
        }},
        {"shader_contract", subtitle_shader_contract_json(identity, {
          {
            {"source_file", "host_sbs_ocr_preprocess_cs.hlsl"},
            {"entrypoint", "main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_ocr_boxes_cs.hlsl"},
            {"entrypoint", "cells_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_ocr_boxes_cs.hlsl"},
            {"entrypoint", "resolve_main"},
            {"target", "cs_5_0"},
          },
        })},
      };
    }

    nlohmann::json subtitle_locator_resolver_contract_json(
      const models::parallax_v2_shader_provenance_t &identity
    ) {
      return {
        {"state_schema", subtitle_locator_state_schema},
        {"state_tag", subtitle_locator_state_tag},
        {"state_word_count", subtitle_locator_state_word_count},
        {"rectangle_capacity", subtitle_locator_rectangle_capacity},
        {"qualification_policy", {
          {"corner_filter_applies_to", "non-ribbon-ordinary-cores"},
          {"corner_edge_clearance",
           "strictly-less-than-floor-content-width-over-divisor"},
          {"corner_edge_divisor", subtitle_locator_corner_edge_divisor},
          {"corner_bottom", "at-or-below-dynamic-roi-bottom-minus-rows"},
          {"corner_bottom_rows", subtitle_locator_corner_bottom_rows},
          {"edge_threshold_equality", "accepted"},
          {"ribbon_exempt", true},
        }},
        {"provisional_current_policy", {
          {"flag", subtitle_locator_provisional_current_flag},
          {"target_word", subtitle_locator_provisional_target_word},
          {"fade_word", subtitle_locator_provisional_fade_word},
          {"applies_to", "first-distinct-unmatched-single-ordinary-replacement"},
          {"prior_requires", {
            {"owner_count", 1u}, {"current_count", 1u}, {"pending_count", 0u},
            {"fade", 2u}, {"event", "none"}, {"unreliable_holds", 0u},
          }},
          {"minimum_vertical_overlap", {
            {"numerator", subtitle_locator_provisional_min_vertical_overlap_numerator},
            {"denominator", subtitle_locator_provisional_min_vertical_overlap_denominator},
          }},
          {"maximum_height_ratio", subtitle_locator_provisional_max_height_ratio},
          {"maximum_center_y_delta", {
            {"numerator",
             subtitle_locator_provisional_max_center_y_delta_shorter_height},
            {"denominator", 1u}, {"unit", "shorter-height"},
          }},
          {"horizontal_center_containment", "mutual-half-open"},
          {"current_authority", "exact-same-frame-selected-ocr8-core-cover-pair"},
          {"durable_state", "owner-generation-target-fade-unchanged"},
          {"duplicate_requires", "exact-identity-core-and-cover"},
          {"hard_cut_allowed", false},
        }},
        {"target_policy", {
          {"units", "binocular-source-pixels"},
          {"placement", {
            {"primary", "aggregate-owner-median-member-center"},
            {"fallback_on_primary_failure", true},
            {"fallback_span",
             "ordinary-core-horizontal-bounds-else-owner-core-horizontal-bounds"},
            {"fallback_top", "ordinary-core-top-else-owner-core-top"},
            {"fallback_step_denominator", subtitle_target_horizontal_step_denominator},
            {"fallback_max_radius_steps",
             subtitle_target_horizontal_fallback_max_radius_steps},
            {"fallback_order_within_radius", {"negative", "positive"}},
            {"fallback_radius_policy", "first-reliable-radius"},
            {"fallback_requires_unclamped_sample_strip", true},
            {"fallback_minimum_coherent_rows", 2u},
            {"fallback_row_median_delta_max",
             subtitle_target_max_row_median_delta_binocular_source_pixels},
            {"fallback_probe_target", "mean-medians"},
            {"fallback_pair_target_delta_max",
             subtitle_target_max_row_median_delta_binocular_source_pixels},
            {"fallback_pair_conflict", "unreliable-stop-search"},
            {"fallback_within_radius_policy", "maximum-mean-within-delta"},
            {"ribbon_places_fallback_with_ordinary", false},
          }},
          {"selection", {
            {"applies_to", "primary"},
            {"samples_per_row", 16u},
            {"median_indices", {7u, 8u}},
            {"iqr_lower_indices", {3u, 4u}},
            {"iqr_upper_indices", {11u, 12u}},
            {"row_validity", "independent-finite-direct-container"},
            {"both_valid_row_iqr", "ignored"},
            {"single_valid_row", "median-if-iqr-at-most-row-iqr-max"},
            {"both_valid_within_delta", "mean-medians"},
            {"both_valid_beyond_delta", "maximum-median"},
          }},
          {"evidence", {
            {"row_iqr_max",
             subtitle_target_max_row_iqr_binocular_source_pixels},
            {"row_median_delta_max",
             subtitle_target_max_row_median_delta_binocular_source_pixels},
          }},
          {"deadband", subtitle_target_deadband_binocular_source_pixels},
          {"ema_alpha", subtitle_target_ema_alpha},
          {"maximum_slew", subtitle_target_max_slew_binocular_source_pixels},
          {"maximum_residual", subtitle_target_max_residual_binocular_source_pixels},
          {"unreliable_hold", {
            {"owner_state_word", 25u},
            {"maximum_distinct_observations", subtitle_target_max_unreliable_holds},
            {"increment_requires",
             "continuing-same-scene-owner-current-authority-valid-target"},
            {"preserve_without_current_authority", true},
            {"duplicate_observation_ages", false},
            {"hard_cut_allowed", false},
          }},
          {"representation_limit", "direct-parallax-container"},
        }},
        {"shader_contract", subtitle_shader_contract_json(identity, {
          {
            {"source_file", "host_sbs_subtitle_locator_cs.hlsl"},
            {"entrypoint", "resolve_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_locator_cs.hlsl"},
            {"entrypoint", "condition_prepare_main"},
            {"target", "cs_5_0"},
          },
          {
            {"source_file", "host_sbs_subtitle_locator_cs.hlsl"},
            {"entrypoint", "condition_main"},
            {"target", "cs_5_0"},
          },
        })},
      };
    }

    bool subtitle_target_is_representable(const float target) {
      // SLR13 tracks a signed local supporting plane. The evidence gates belong to the shader
      // observation transaction; the compact state only carries an accepted target, so native
      // dump trust authenticates its finite V2 representation domain exactly.
      return std::isfinite(target) && std::abs(target) <= direct_container_limit;
    }

    std::uint32_t subtitle_word(
      const std::vector<std::uint8_t> &bytes,
      const std::size_t index
    ) {
      std::uint32_t value = 0u;
      std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
      return value;
    }

    std::uint64_t subtitle_u64(
      const std::vector<std::uint8_t> &bytes,
      const std::size_t low_index
    ) {
      return static_cast<std::uint64_t>(subtitle_word(bytes, low_index)) |
             (static_cast<std::uint64_t>(subtitle_word(bytes, low_index + 1u)) << 32u);
    }

    bool subtitle_ocr_boxes_are_canonical(
      const std::vector<std::uint8_t> &bytes,
      const std::size_t offset,
      const std::uint32_t count,
      const std::uint32_t capacity,
      const std::uint32_t field_width,
      const std::uint32_t field_height,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_top,
      const std::uint32_t roi_bottom,
      const std::uint32_t ribbon_min_bottom,
      const bool final_boxes
    ) {
      for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
        const std::size_t first =
          offset + static_cast<std::size_t>(slot) * subtitle_ocr_box_word_count;
        if (slot >= count) {
          for (std::size_t word = 0u; word < subtitle_ocr_box_word_count; ++word) {
            if (subtitle_word(bytes, first + word) != 0u) {
              return false;
            }
          }
          continue;
        }
        const auto left = subtitle_word(bytes, first);
        const auto top = subtitle_word(bytes, first + 1u);
        const auto right = subtitle_word(bytes, first + 2u);
        const auto bottom = subtitle_word(bytes, first + 3u);
        const float score = std::bit_cast<float>(subtitle_word(bytes, first + 4u));
        const auto box_flags = subtitle_word(bytes, first + 5u);
        const auto island_count = subtitle_word(bytes, first + 6u);
        const auto structural_gap_count = subtitle_word(bytes, first + 7u);
        const bool ribbon = (box_flags & subtitle_ocr_box_flag_ribbon) != 0u;
        if (left >= right || top >= bottom || left < tensor_content.left ||
            top < tensor_content.top || right > tensor_content.right ||
            bottom > tensor_content.bottom || right > field_width ||
            bottom > field_height || top < roi_top ||
            !std::isfinite(score) || score < subtitle_ocr_min_mean_score ||
            score > 1.0f ||
            (box_flags & ~subtitle_ocr_box_known_flags) != 0u ||
            island_count == 0u || island_count > subtitle_ocr_output_width ||
            structural_gap_count >= island_count) {
          return false;
        }
        if (ribbon) {
          if (island_count < subtitle_ocr_ribbon_min_structural_gaps + 1u ||
              structural_gap_count < subtitle_ocr_ribbon_min_structural_gaps) {
            return false;
          }
          if (final_boxes) {
            if (left != tensor_content.left || right != tensor_content.right ||
                bottom != tensor_content.bottom) {
              return false;
            }
          } else if (
            bottom < ribbon_min_bottom || bottom > roi_bottom ||
            static_cast<std::uint64_t>(right - left) *
                subtitle_ocr_ribbon_min_width_denominator <
              static_cast<std::uint64_t>(tensor_content.width()) *
                subtitle_ocr_ribbon_min_width_numerator
          ) {
            return false;
          }
        } else if (bottom > roi_bottom) {
          return false;
        }
      }
      return true;
    }

    bool subtitle_ocr_pairs_are_canonical(
      const std::vector<std::uint8_t> &ocr,
      const std::uint32_t count
    ) {
      for (std::uint32_t slot = 0u; slot < count; ++slot) {
        const std::size_t raw = subtitle_ocr_raw_box_offset +
          static_cast<std::size_t>(slot) * subtitle_ocr_box_word_count;
        const std::size_t final = subtitle_ocr_final_box_offset +
          static_cast<std::size_t>(slot) * subtitle_ocr_box_word_count;
        if (subtitle_word(ocr, final) > subtitle_word(ocr, raw) ||
            subtitle_word(ocr, final + 1u) > subtitle_word(ocr, raw + 1u) ||
            subtitle_word(ocr, final + 2u) < subtitle_word(ocr, raw + 2u) ||
            subtitle_word(ocr, final + 3u) < subtitle_word(ocr, raw + 3u)) {
          return false;
        }
        for (std::size_t word = 4u; word < subtitle_ocr_box_word_count; ++word) {
          if (subtitle_word(ocr, raw + word) != subtitle_word(ocr, final + word)) {
            return false;
          }
        }
      }
      return true;
    }

    struct subtitle_rectangle_summary_t {
      std::array<std::uint32_t, 4> bbox {};
      std::uint32_t area = 0u;
    };

    bool subtitle_locator_rectangles_are_canonical(
      const std::vector<std::uint8_t> &bytes,
      const std::size_t offset,
      const std::uint32_t count,
      const std::uint32_t field_width,
      const std::uint32_t field_height,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_top,
      const std::uint32_t roi_bottom,
      const std::uint32_t ribbon_min_bottom,
      const std::uint32_t ribbon_mask,
      const bool current_cover,
      subtitle_rectangle_summary_t &summary
    ) {
      summary = {};
      for (std::uint32_t slot = 0u; slot < subtitle_locator_rectangle_capacity; ++slot) {
        const std::size_t first = offset + static_cast<std::size_t>(slot) * 4u;
        const auto left = subtitle_word(bytes, first);
        const auto top = subtitle_word(bytes, first + 1u);
        const auto right = subtitle_word(bytes, first + 2u);
        const auto bottom = subtitle_word(bytes, first + 3u);
        const bool ribbon = (ribbon_mask & (1u << slot)) != 0u;
        if (slot >= count) {
          if (left != 0u || top != 0u || right != 0u || bottom != 0u) {
            return false;
          }
        } else if (left >= right || top >= bottom ||
                   left < tensor_content.left || top < tensor_content.top ||
                   right > tensor_content.right || bottom > tensor_content.bottom ||
                   right > field_width || bottom > field_height || top < roi_top ||
                   (current_cover && ribbon ?
                      (left != tensor_content.left || right != tensor_content.right ||
                       bottom != tensor_content.bottom) :
                      (bottom > roi_bottom || (ribbon && bottom < ribbon_min_bottom)))) {
          return false;
        } else {
          if (!current_cover && slot != 0u) {
            const std::size_t previous = first - 4u;
            const auto previous_left = subtitle_word(bytes, previous);
            const auto previous_top = subtitle_word(bytes, previous + 1u);
            if (top < previous_top || (top == previous_top && left < previous_left)) {
              return false;
            }
          }
          if (slot == 0u) {
            summary.bbox = {left, top, right, bottom};
          } else {
            summary.bbox[0] = std::min(summary.bbox[0], left);
            summary.bbox[1] = std::min(summary.bbox[1], top);
            summary.bbox[2] = std::max(summary.bbox[2], right);
            summary.bbox[3] = std::max(summary.bbox[3], bottom);
          }
          summary.area += (right - left) * (bottom - top);
        }
      }
      return true;
    }

    bool subtitle_rectangle_summary_matches(
      const std::vector<std::uint8_t> &bytes,
      const std::size_t bbox_offset,
      const std::size_t area_offset,
      const subtitle_rectangle_summary_t &summary
    ) {
      return subtitle_word(bytes, bbox_offset) == summary.bbox[0] &&
             subtitle_word(bytes, bbox_offset + 1u) == summary.bbox[1] &&
             subtitle_word(bytes, bbox_offset + 2u) == summary.bbox[2] &&
             subtitle_word(bytes, bbox_offset + 3u) == summary.bbox[3] &&
             subtitle_word(bytes, area_offset) == summary.area;
    }

    struct subtitle_selected_box_t {
      std::array<std::uint32_t, 4> core {};
      std::array<std::uint32_t, 4> cover {};
      bool ribbon = false;
    };

    std::uint32_t subtitle_rectangle_area(
      const std::array<std::uint32_t, 4> &rectangle
    ) {
      return (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1]);
    }

    bool subtitle_lines_are_coherent(
      const std::array<std::uint32_t, 4> &a,
      const std::array<std::uint32_t, 4> &b
    ) {
      const auto width_a = a[2] - a[0];
      const auto width_b = b[2] - b[0];
      const auto height_a = a[3] - a[1];
      const auto height_b = b[3] - b[1];
      const auto overlap = std::min(a[2], b[2]) > std::max(a[0], b[0]) ?
        std::min(a[2], b[2]) - std::max(a[0], b[0]) : 0u;
      const auto center_x_a = a[0] + a[2];
      const auto center_x_b = b[0] + b[2];
      const auto center_x_delta_twice = center_x_a > center_x_b ?
        center_x_a - center_x_b : center_x_b - center_x_a;
      const auto left_delta = a[0] > b[0] ? a[0] - b[0] : b[0] - a[0];
      const auto right_delta = a[2] > b[2] ? a[2] - b[2] : b[2] - a[2];
      const auto center_y_a = a[1] + a[3];
      const auto center_y_b = b[1] + b[3];
      const auto center_y_delta_twice = center_y_a > center_y_b ?
        center_y_a - center_y_b : center_y_b - center_y_a;
      const auto gap = a[3] <= b[1] ? b[1] - a[3] :
                       b[3] <= a[1] ? a[1] - b[3] : 0u;
      return static_cast<std::uint64_t>(overlap) * 2u >= std::min(width_a, width_b) &&
             (center_x_delta_twice <= std::max(height_a, height_b) ||
              left_delta <= std::max(height_a, height_b) ||
              right_delta <= std::max(height_a, height_b)) &&
             std::max(height_a, height_b) <= 2u * std::min(height_a, height_b) &&
             center_y_delta_twice >= std::min(height_a, height_b) &&
             static_cast<std::uint64_t>(gap) * 2u <= std::max(height_a, height_b);
    }

    bool subtitle_segments_share_baseline(
      const std::array<std::uint32_t, 4> &a,
      const std::array<std::uint32_t, 4> &b,
      const std::uint32_t content_width
    ) {
      const bool a_before_b = a[2] <= b[0];
      const bool b_before_a = b[2] <= a[0];
      if (!a_before_b && !b_before_a) {
        return false;
      }
      const auto height_a = a[3] - a[1];
      const auto height_b = b[3] - b[1];
      const auto shorter_height = std::min(height_a, height_b);
      const auto taller_height = std::max(height_a, height_b);
      const auto vertical_overlap = std::min(a[3], b[3]) > std::max(a[1], b[1]) ?
        std::min(a[3], b[3]) - std::max(a[1], b[1]) : 0u;
      const auto center_y_a = a[1] + a[3];
      const auto center_y_b = b[1] + b[3];
      const auto center_y_delta_twice = center_y_a > center_y_b ?
        center_y_a - center_y_b : center_y_b - center_y_a;
      const auto horizontal_gap = a_before_b ? b[0] - a[2] : a[0] - b[2];
      const auto combined_span = std::max(a[2], b[2]) - std::min(a[0], b[0]);
      const auto maximum_width = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(content_width) *
          subtitle_locator_max_width_numerator /
          subtitle_locator_max_width_denominator
      );
      return static_cast<std::uint64_t>(vertical_overlap) * 4u >=
               static_cast<std::uint64_t>(shorter_height) * 3u &&
             taller_height <= 2u * shorter_height &&
             center_y_delta_twice <= shorter_height &&
             horizontal_gap <= 8u * taller_height &&
             combined_span <= maximum_width;
    }

    std::vector<subtitle_selected_box_t> subtitle_selected_ocr_boxes(
      const std::vector<std::uint8_t> &ocr,
      const std::uint32_t final_count,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_bottom
    ) {
      const auto content_width = tensor_content.width();
      std::vector<subtitle_selected_box_t> qualified;
      qualified.reserve(final_count);
      std::uint32_t ribbon_mask = 0u;
      const auto maximum_width = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(content_width) *
          subtitle_locator_max_width_numerator /
          subtitle_locator_max_width_denominator
      );
      for (std::uint32_t slot = 0u; slot < final_count; ++slot) {
        const std::size_t raw = subtitle_ocr_raw_box_offset +
          static_cast<std::size_t>(slot) * subtitle_ocr_box_word_count;
        const std::size_t final = subtitle_ocr_final_box_offset +
          static_cast<std::size_t>(slot) * subtitle_ocr_box_word_count;
        subtitle_selected_box_t box {
          .core = {
            subtitle_word(ocr, raw), subtitle_word(ocr, raw + 1u),
            subtitle_word(ocr, raw + 2u), subtitle_word(ocr, raw + 3u),
          },
          .cover = {
            subtitle_word(ocr, final), subtitle_word(ocr, final + 1u),
            subtitle_word(ocr, final + 2u), subtitle_word(ocr, final + 3u),
          },
          .ribbon = (subtitle_word(ocr, raw + 5u) & subtitle_ocr_box_flag_ribbon) != 0u,
        };
        const auto width = box.core[2] - box.core[0];
        const auto height = box.core[3] - box.core[1];
        const auto corner_edge_threshold =
          content_width / subtitle_locator_corner_edge_divisor;
        const auto left_clearance = box.core[0] - tensor_content.left;
        const auto right_clearance = tensor_content.right - box.core[2];
        const bool bottom_corner_ordinary =
          !box.ribbon && std::min(left_clearance, right_clearance) < corner_edge_threshold &&
          box.core[3] + subtitle_locator_corner_bottom_rows >= roi_bottom;
        if (width < subtitle_locator_min_width_cells ||
            bottom_corner_ordinary ||
            (!box.ribbon && width > maximum_width) ||
            height < subtitle_locator_min_height_cells ||
            static_cast<std::uint64_t>(width) *
                subtitle_locator_min_aspect_denominator <
              static_cast<std::uint64_t>(subtitle_locator_min_aspect_numerator) * height) {
          continue;
        }
        if (box.ribbon) {
          ribbon_mask |= 1u << qualified.size();
        }
        qualified.push_back(box);
      }
      if (qualified.empty()) {
        return {};
      }

      std::array<std::uint32_t, subtitle_ocr_final_box_capacity> component_masks {};
      for (std::size_t index = 0u; index < qualified.size(); ++index) {
        if (!qualified[index].ribbon) {
          component_masks[index] = 1u << index;
        }
      }
      for (std::uint32_t pass = 0u; pass < subtitle_ocr_final_box_capacity; ++pass) {
        for (std::size_t a = 0u; a < qualified.size(); ++a) {
          auto expanded = component_masks[a];
          for (std::size_t b = 0u; b < qualified.size(); ++b) {
            if (qualified[b].ribbon || (expanded & (1u << b)) == 0u) {
              continue;
            }
            for (std::size_t c = 0u; c < qualified.size(); ++c) {
              if (!qualified[c].ribbon &&
                  (subtitle_lines_are_coherent(qualified[b].core, qualified[c].core) ||
                   subtitle_segments_share_baseline(
                     qualified[b].core, qualified[c].core, content_width
                   ))) {
                expanded |= 1u << c;
              }
            }
          }
          component_masks[a] = expanded;
        }
      }

      std::uint32_t best_mask = 0u;
      std::uint32_t best_area = 0u;
      std::array<std::uint32_t, 4> best_bbox {};
      for (std::size_t root = 0u; root < qualified.size(); ++root) {
        if (qualified[root].ribbon) {
          continue;
        }
        const auto mask = component_masks[root];
        if ((mask & ((1u << root) - 1u)) != 0u) {
          continue;
        }
        std::uint32_t count = 0u;
        std::uint32_t area = 0u;
        std::array<std::uint32_t, 4> bbox {
          std::numeric_limits<std::uint32_t>::max(),
          std::numeric_limits<std::uint32_t>::max(), 0u, 0u
        };
        for (std::size_t index = 0u; index < qualified.size(); ++index) {
          if ((mask & (1u << index)) == 0u) {
            continue;
          }
          const auto &rectangle = qualified[index].core;
          ++count;
          area += subtitle_rectangle_area(rectangle);
          bbox[0] = std::min(bbox[0], rectangle[0]);
          bbox[1] = std::min(bbox[1], rectangle[1]);
          bbox[2] = std::max(bbox[2], rectangle[2]);
          bbox[3] = std::max(bbox[3], rectangle[3]);
        }
        const bool better = count <= subtitle_locator_rectangle_capacity &&
          bbox[2] - bbox[0] <= maximum_width &&
          (best_mask == 0u || area > best_area ||
           (area == best_area &&
            (bbox[3] > best_bbox[3] ||
             (bbox[3] == best_bbox[3] &&
              (bbox[1] > best_bbox[1] ||
               (bbox[1] == best_bbox[1] && bbox[0] < best_bbox[0]))))));
        if (better) {
          best_mask = mask;
          best_area = area;
          best_bbox = bbox;
        }
      }

      const auto selected_mask = best_mask | ribbon_mask;
      if (selected_mask == 0u ||
          std::popcount(selected_mask) > subtitle_locator_rectangle_capacity) {
        return {};
      }
      std::vector<subtitle_selected_box_t> selected;
      for (std::size_t index = 0u; index < qualified.size(); ++index) {
        if ((selected_mask & (1u << index)) != 0u) {
          selected.push_back(qualified[index]);
        }
      }
      std::stable_sort(selected.begin(), selected.end(), [](const auto &a, const auto &b) {
        return a.core[1] != b.core[1] ? a.core[1] < b.core[1] : a.core[0] < b.core[0];
      });
      return selected;
    }

    bool subtitle_current_rectangles_match_ocr_selection(
      const std::vector<std::uint8_t> &ocr,
      const std::uint32_t final_count,
      const std::vector<std::uint8_t> &locator,
      const std::uint32_t current_count,
      const std::uint32_t current_ribbon_mask,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_bottom
    ) {
      const auto selected = subtitle_selected_ocr_boxes(
        ocr, final_count, tensor_content, roi_bottom
      );
      std::size_t selected_position = 0u;
      for (std::uint32_t current_slot = 0u; current_slot < current_count; ++current_slot) {
        const std::size_t current =
          subtitle_locator_current_offset + static_cast<std::size_t>(current_slot) * 4u;
        const bool current_ribbon =
          (current_ribbon_mask & (1u << current_slot)) != 0u;
        bool found = false;
        while (selected_position < selected.size()) {
          const auto &candidate = selected[selected_position++];
          if (subtitle_word(locator, current) == candidate.cover[0] &&
              subtitle_word(locator, current + 1u) == candidate.cover[1] &&
              subtitle_word(locator, current + 2u) == candidate.cover[2] &&
              subtitle_word(locator, current + 3u) == candidate.cover[3] &&
              current_ribbon == candidate.ribbon) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    }

    bool subtitle_provisional_current_matches_ocr_selection(
      const std::vector<std::uint8_t> &ocr,
      const std::uint32_t final_count,
      const std::vector<std::uint8_t> &locator,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_bottom
    ) {
      const auto selected = subtitle_selected_ocr_boxes(
        ocr, final_count, tensor_content, roi_bottom
      );
      if (selected.size() != 1u || selected.front().ribbon) {
        return false;
      }
      const auto &candidate = selected.front();
      for (std::size_t coordinate = 0u; coordinate < 4u; ++coordinate) {
        if (subtitle_word(
              locator, subtitle_locator_pending_offset + coordinate
            ) != candidate.core[coordinate] ||
            subtitle_word(
              locator, subtitle_locator_current_offset + coordinate
            ) != candidate.cover[coordinate]) {
          return false;
        }
      }
      return true;
    }

    bool subtitle_provisional_geometry_is_canonical(
      const std::array<std::uint32_t, 4u> &owner,
      const std::array<std::uint32_t, 4u> &pending
    ) {
      const std::uint64_t owner_height = owner[3] - owner[1];
      const std::uint64_t pending_height = pending[3] - pending[1];
      const auto shorter_height = std::min(owner_height, pending_height);
      const auto taller_height = std::max(owner_height, pending_height);
      const std::uint64_t overlap_top = std::max(owner[1], pending[1]);
      const std::uint64_t overlap_bottom = std::min(owner[3], pending[3]);
      const std::uint64_t vertical_overlap =
        overlap_bottom > overlap_top ? overlap_bottom - overlap_top : 0u;
      const std::uint64_t owner_center_x_twice =
        static_cast<std::uint64_t>(owner[0]) + owner[2];
      const std::uint64_t pending_center_x_twice =
        static_cast<std::uint64_t>(pending[0]) + pending[2];
      const std::uint64_t owner_center_y_twice =
        static_cast<std::uint64_t>(owner[1]) + owner[3];
      const std::uint64_t pending_center_y_twice =
        static_cast<std::uint64_t>(pending[1]) + pending[3];
      const auto center_y_delta_twice = owner_center_y_twice > pending_center_y_twice ?
        owner_center_y_twice - pending_center_y_twice :
        pending_center_y_twice - owner_center_y_twice;
      const std::uint64_t intersection_left = std::max(owner[0], pending[0]);
      const std::uint64_t intersection_right = std::min(owner[2], pending[2]);
      const std::uint64_t intersection_width =
        intersection_right > intersection_left ? intersection_right - intersection_left : 0u;
      const std::uint64_t intersection = intersection_width * vertical_overlap;
      const std::uint64_t owner_area =
        static_cast<std::uint64_t>(owner[2] - owner[0]) * owner_height;
      const std::uint64_t pending_area =
        static_cast<std::uint64_t>(pending[2] - pending[0]) * pending_height;
      const std::uint64_t union_area = owner_area + pending_area - intersection;
      const float iou = union_area == 0u ? 0.0f :
        static_cast<float>(intersection) / static_cast<float>(union_area);
      return iou < subtitle_locator_match_iou_threshold &&
             vertical_overlap *
                 subtitle_locator_provisional_min_vertical_overlap_denominator >=
               shorter_height *
                 subtitle_locator_provisional_min_vertical_overlap_numerator &&
             taller_height <=
               subtitle_locator_provisional_max_height_ratio * shorter_height &&
             center_y_delta_twice <=
               subtitle_locator_provisional_max_center_y_delta_shorter_height *
                 shorter_height &&
             owner_center_x_twice >= 2u * pending[0] &&
             owner_center_x_twice < 2u * pending[2] &&
             pending_center_x_twice >= 2u * owner[0] &&
             pending_center_x_twice < 2u * owner[2];
    }

    bool subtitle_provisional_geometry_is_canonical(
      const std::vector<std::uint8_t> &locator
    ) {
      const auto rectangle = [&locator](const std::size_t offset) {
        return std::array<std::uint32_t, 4u> {
          subtitle_word(locator, offset + 0u), subtitle_word(locator, offset + 1u),
          subtitle_word(locator, offset + 2u), subtitle_word(locator, offset + 3u),
        };
      };
      return subtitle_provisional_geometry_is_canonical(
        rectangle(subtitle_locator_owner_offset),
        rectangle(subtitle_locator_pending_offset)
      );
    }

    bool subtitle_locator_state_is_canonical(
      const std::vector<std::uint8_t> &ocr,
      const std::uint32_t ocr_flags,
      const std::uint32_t final_count,
      const std::vector<std::uint8_t> &locator,
      const std::uint32_t field_width,
      const std::uint32_t field_height,
      const models::depth_tensor_content_rect_t tensor_content,
      const std::uint32_t roi_top,
      const std::uint32_t roi_bottom,
      const std::uint32_t ribbon_min_bottom
    ) {
      const auto flags = subtitle_word(locator, 2u);
      const auto owner_generation = subtitle_word(locator, 3u);
      const auto owner_count = subtitle_word(locator, 4u);
      const auto pending_count = subtitle_word(locator, 12u);
      const auto target_bits = subtitle_word(locator, 18u);
      const float target = std::bit_cast<float>(target_bits);
      const auto target_generation = subtitle_word(locator, 19u);
      const auto current_count = subtitle_word(locator, 20u);
      const auto last_event = subtitle_word(locator, 21u);
      const auto fade = subtitle_word(locator, 24u);
      const auto hold_or_grace = subtitle_word(locator, 25u);
      const auto packed_grace_x = subtitle_word(locator, 29u);
      const auto packed_grace_y = subtitle_word(locator, 30u);
      const auto packed_kinds = subtitle_word(locator, subtitle_locator_kind_word);
      const auto owner_ribbon_mask =
        (packed_kinds >> subtitle_locator_owner_kind_shift) & subtitle_locator_kind_mask;
      const auto pending_ribbon_mask =
        (packed_kinds >> subtitle_locator_pending_kind_shift) & subtitle_locator_kind_mask;
      const auto current_ribbon_mask =
        (packed_kinds >> subtitle_locator_current_kind_shift) & subtitle_locator_kind_mask;
      const auto known_kind_bits =
        (subtitle_locator_kind_mask << subtitle_locator_owner_kind_shift) |
        (subtitle_locator_kind_mask << subtitle_locator_pending_kind_shift) |
        (subtitle_locator_kind_mask << subtitle_locator_current_kind_shift);
      const auto count_mask = [](const std::uint32_t count) {
        return count == 0u ? 0u : (1u << count) - 1u;
      };
      const bool owner = (flags & subtitle_locator_flag_owner) != 0u;
      const bool pending = (flags & subtitle_locator_flag_pending) != 0u;
      const bool target_valid = (flags & subtitle_locator_flag_target_valid) != 0u;
      const bool target_reset = (flags & subtitle_locator_flag_target_reset) != 0u;
      const bool provisional_current =
        (flags & subtitle_locator_flag_provisional_current) != 0u;
      const bool target_in_range = subtitle_target_is_representable(target);
      const auto provisional_target_bits = subtitle_word(
        locator, subtitle_locator_provisional_target_word
      );
      const float provisional_target = std::bit_cast<float>(provisional_target_bits);
      const auto provisional_fade = subtitle_word(
        locator, subtitle_locator_provisional_fade_word
      );

      subtitle_rectangle_summary_t owner_summary;
      subtitle_rectangle_summary_t pending_summary;
      subtitle_rectangle_summary_t current_summary;
      if ((flags & ~subtitle_locator_known_flags) != 0u ||
          owner_count > subtitle_locator_rectangle_capacity ||
          pending_count > subtitle_locator_rectangle_capacity ||
          current_count > subtitle_locator_rectangle_capacity ||
          (packed_kinds & ~known_kind_bits) != 0u ||
          (owner_ribbon_mask & ~count_mask(owner_count)) != 0u ||
          (pending_ribbon_mask & ~count_mask(pending_count)) != 0u ||
          (current_ribbon_mask & ~count_mask(current_count)) != 0u ||
          fade > subtitle_locator_max_fade || last_event > subtitle_locator_max_event ||
          !subtitle_locator_rectangles_are_canonical(
            locator, subtitle_locator_owner_offset, owner_count,
            field_width, field_height, tensor_content, roi_top, roi_bottom,
            ribbon_min_bottom,
            owner_ribbon_mask, false, owner_summary
          ) ||
          !subtitle_locator_rectangles_are_canonical(
            locator, subtitle_locator_pending_offset, pending_count,
            field_width, field_height, tensor_content, roi_top, roi_bottom,
            ribbon_min_bottom,
            pending_ribbon_mask, false, pending_summary
          ) ||
          !subtitle_locator_rectangles_are_canonical(
            locator, subtitle_locator_current_offset, current_count,
            field_width, field_height, tensor_content, roi_top, roi_bottom,
            ribbon_min_bottom,
            current_ribbon_mask, true, current_summary
          ) ||
          !subtitle_rectangle_summary_matches(locator, 5u, 9u, owner_summary) ||
          !subtitle_rectangle_summary_matches(locator, 13u, 17u, pending_summary) ||
          owner != (owner_count != 0u) || pending != (pending_count != 0u) ||
          owner != (owner_generation != 0u) ||
          current_count > owner_count ||
          (current_count != 0u && !provisional_current &&
           (!owner || !target_valid || fade == 0u)) ||
          (ocr_flags == 0u && current_count != 0u) ||
          !subtitle_current_rectangles_match_ocr_selection(
            ocr, final_count, locator, current_count, current_ribbon_mask,
            tensor_content, roi_bottom
          )) {
        return false;
      }

      if (target_valid) {
        if (!owner || target_reset || target_generation != owner_generation ||
            !target_in_range || fade == 0u) {
          return false;
        }
      } else if (target_reset) {
        if (!owner || target_bits != 0u || target_generation != 0u ||
            current_count != 0u || fade != 0u) {
          return false;
        }
      }

      if (provisional_current) {
        if (!pending || target_reset || pending_count != 1u || current_count != 1u ||
            owner_ribbon_mask != 0u || pending_ribbon_mask != 0u ||
            current_ribbon_mask != 0u || last_event != 0u ||
            hold_or_grace != 0u ||
            !subtitle_target_is_representable(provisional_target) ||
            !subtitle_provisional_current_matches_ocr_selection(
              ocr, final_count, locator, tensor_content, roi_bottom
            )) {
          return false;
        }
        if (!owner || !target_valid || owner_count != 1u || fade != 2u ||
            (provisional_fade != 1u && provisional_fade != 2u) ||
            !subtitle_provisional_geometry_is_canonical(locator)) {
          return false;
        }
      }

      const bool packed_grace_zero = packed_grace_x == 0u && packed_grace_y == 0u;
      if (owner) {
        if (hold_or_grace > subtitle_target_max_unreliable_holds ||
            (!provisional_current && !packed_grace_zero) ||
            (hold_or_grace != 0u &&
             (!target_valid || fade == 0u || last_event != 0u))) {
          return false;
        }
        if (!target_valid && !target_reset &&
            (target_bits != 0u || target_generation != 0u ||
             current_count != 0u || fade != 0u)) {
          return false;
        }
      } else if (hold_or_grace == 0u) {
        if (provisional_current || target_bits != 0u || target_generation != 0u ||
            !packed_grace_zero ||
            current_count != 0u || target_valid || target_reset || fade != 0u) {
          return false;
        }
      } else {
        const auto grace_left = packed_grace_x & 0xFFFFu;
        const auto grace_right = packed_grace_x >> 16u;
        const auto grace_top = packed_grace_y & 0xFFFFu;
        const auto grace_bottom = packed_grace_y >> 16u;
        if (provisional_current ||
            hold_or_grace > subtitle_locator_death_grace_observations ||
            target_valid || target_reset || target_generation != 0u || !target_in_range ||
            current_count != 0u || fade != 0u ||
            grace_left >= grace_right || grace_top >= grace_bottom ||
            grace_left < tensor_content.left || grace_right > tensor_content.right ||
            grace_bottom > tensor_content.bottom || grace_top < roi_top ||
            grace_bottom > roi_bottom) {
          return false;
        }
      }
      return true;
    }

  }  // namespace

  namespace detail {

    bool subtitle_ocr_record_is_canonical_for_frame(
      const std::vector<std::uint8_t> &ocr,
      const frame &completed
    ) {
      if (ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t)) {
        return false;
      }
      const auto field_width = static_cast<std::uint32_t>(completed.model_width);
      const auto field_height = static_cast<std::uint32_t>(completed.model_height);
      const auto geometry = models::fit_subtitle_analysis_geometry(
        completed.depth_input_region.width(),
        completed.depth_input_region.height(),
        {completed.model_width, completed.model_height},
        completed.depth_input_region.tensor_content
      );
      if (!geometry.valid()) {
        return false;
      }
      const auto tensor_content = geometry.tensor_content;
      const auto roi_top = geometry.roi_top;
      const auto roi_bottom = geometry.roi_bottom;
      const auto ribbon_min_bottom = geometry.ribbon_min_bottom;
      const auto ocr_flags = subtitle_word(ocr, 2u);
      const auto raw_count = subtitle_word(ocr, 3u);
      const auto final_count = subtitle_word(ocr, 4u);
      if (subtitle_word(ocr, 0u) != subtitle_ocr_record_schema ||
          subtitle_word(ocr, 1u) != subtitle_ocr_record_tag ||
          ocr_flags > 1u || raw_count > subtitle_ocr_raw_box_capacity ||
          final_count > subtitle_ocr_final_box_capacity ||
          final_count != raw_count ||
          (ocr_flags == 0u && (raw_count != 0u || final_count != 0u)) ||
          subtitle_u64(ocr, 5u) != completed.matched_frame_id ||
          subtitle_u64(ocr, 7u) != completed.depth_input_region.analysis_generation ||
          subtitle_word(ocr, 9u) != completed.depth_input_region.width() ||
          subtitle_word(ocr, 10u) != completed.depth_input_region.height() ||
          subtitle_word(ocr, 11u) != field_width ||
          subtitle_word(ocr, 12u) != field_height ||
          subtitle_word(ocr, 13u) != roi_top ||
          subtitle_word(ocr, 14u) != roi_bottom ||
          subtitle_word(ocr, 15u) != 0u ||
          !subtitle_ocr_boxes_are_canonical(
            ocr, subtitle_ocr_raw_box_offset, raw_count, subtitle_ocr_raw_box_capacity,
            field_width, field_height, tensor_content, roi_top, roi_bottom,
            ribbon_min_bottom, false
          ) ||
          !subtitle_ocr_boxes_are_canonical(
            ocr, subtitle_ocr_final_box_offset, final_count, subtitle_ocr_final_box_capacity,
            field_width, field_height, tensor_content, roi_top, roi_bottom,
            ribbon_min_bottom, true
          ) || !subtitle_ocr_pairs_are_canonical(ocr, final_count)) {
        return false;
      }

      return true;
    }

    bool subtitle_records_match_frame(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const frame &completed,
      const std::uint32_t confirmed_cut_count
    ) {
      if (!subtitle_ocr_record_is_canonical_for_frame(ocr, completed) ||
          locator.size() != subtitle_locator_state_word_count * sizeof(std::uint32_t)) {
        return false;
      }
      const auto field_width = static_cast<std::uint32_t>(completed.model_width);
      const auto field_height = static_cast<std::uint32_t>(completed.model_height);
      const auto geometry = models::fit_subtitle_analysis_geometry(
        completed.depth_input_region.width(),
        completed.depth_input_region.height(),
        {completed.model_width, completed.model_height},
        completed.depth_input_region.tensor_content
      );
      if (!geometry.valid()) {
        return false;
      }
      const auto ocr_flags = subtitle_word(ocr, 2u);
      const auto final_count = subtitle_word(ocr, 4u);
      if (subtitle_word(locator, 0u) != subtitle_locator_state_schema ||
          subtitle_word(locator, 1u) != subtitle_locator_state_tag ||
          subtitle_u64(locator, 10u) != completed.depth_input_region.analysis_generation ||
          subtitle_u64(locator, 22u) != completed.matched_frame_id ||
          subtitle_word(locator, 26u) != confirmed_cut_count ||
          subtitle_word(locator, 27u) != field_width ||
          subtitle_word(locator, 28u) != field_height ||
          !subtitle_locator_state_is_canonical(
            ocr, ocr_flags, final_count, locator,
            field_width, field_height, geometry.tensor_content,
            geometry.roi_top, geometry.roi_bottom, geometry.ribbon_min_bottom
          )) {
        return false;
      }
      return true;
    }

    bool gpu_trace_ring_is_canonical(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      using namespace models::host_sbs_gpu_trace;
      if (ring.size() != ring_byte_count || completed.matched_frame_id == 0u ||
          completed.model_width <= 0 || completed.model_height <= 0 ||
          completed.depth_input_region.width() == 0u ||
          completed.depth_input_region.height() == 0u) {
        return false;
      }
      const auto word = [&ring](const std::size_t index) {
        std::uint32_t value = 0u;
        std::memcpy(
          &value,
          ring.data() + index * sizeof(std::uint32_t),
          sizeof(value)
        );
        return value;
      };
      if (word(word_index(header_word_e::schema)) != ring_schema ||
          word(word_index(header_word_e::tag)) != ring_tag ||
          word(word_index(header_word_e::capacity)) != capacity ||
          word(word_index(header_word_e::record_words)) != record_word_count) {
        return false;
      }
      const auto next_sequence = join_u64(
        word(word_index(header_word_e::next_sequence_low)),
        word(word_index(header_word_e::next_sequence_high))
      );
      const auto next_slot = word(word_index(header_word_e::next_slot));
      const auto count = word(word_index(header_word_e::committed_count));
      if (next_sequence == 0u || next_slot >= capacity || count == 0u ||
          count > capacity || next_sequence <= count) {
        return false;
      }
      for (std::size_t index = word_index(header_word_e::reserved_begin);
           index < header_word_count; ++index) {
        if (word(index) != 0u) {
          return false;
        }
      }

      const auto oldest_sequence = next_sequence - count;
      const auto oldest_slot = (next_slot + capacity - count) % capacity;
      bool matched_frame_present = false;
      std::uint64_t previous_observation_timestamp_us = 0u;
      for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        const auto slot = (oldest_slot + ordinal) % capacity;
        const auto base = record_base(slot);
        const auto record = [base, &word](const record_word_e field) {
          return word(base + word_index(field));
        };
        if (record(record_word_e::schema) != ring_schema ||
            record(record_word_e::commit_tag) != record_tag ||
            join_u64(
              record(record_word_e::sequence_low),
              record(record_word_e::sequence_high)
            ) != oldest_sequence + ordinal ||
            record(record_word_e::transaction_words) != transaction_word_count ||
            record(record_word_e::reserved0) != 0u ||
            record(record_word_e::source_width) == 0u ||
            record(record_word_e::source_height) == 0u ||
            record(record_word_e::field_width) == 0u ||
            record(record_word_e::field_height) == 0u) {
          return false;
        }
        const auto observation_timestamp_us = join_u64(
          record(record_word_e::observation_timestamp_low),
          record(record_word_e::observation_timestamp_high)
        );
        if (observation_timestamp_us == 0u ||
            (previous_observation_timestamp_us != 0u &&
             observation_timestamp_us < previous_observation_timestamp_us)) {
          return false;
        }
        previous_observation_timestamp_us = observation_timestamp_us;
        for (std::size_t index = word_index(record_word_e::reserved_begin);
             index < record_word_count; ++index) {
          if (word(base + index) != 0u) {
            return false;
          }
        }
        const auto frame_id = join_u64(
          record(record_word_e::frame_low),
          record(record_word_e::frame_high)
        );
        const auto analysis_generation = join_u64(
          record(record_word_e::analysis_generation_low),
          record(record_word_e::analysis_generation_high)
        );
        const auto domain_tag = join_u64(
          record(record_word_e::domain_tag_low),
          record(record_word_e::domain_tag_high)
        );
        const auto transaction_token = join_u64(
          record(record_word_e::transaction_token_low),
          record(record_word_e::transaction_token_high)
        );
        if (frame_id == 0u || domain_tag == 0u || transaction_token == 0u) {
          return false;
        }
        const bool matched_frame = frame_id == completed.matched_frame_id;

        const auto submission_class = static_cast<submission_class_e>(
          record(record_word_e::submission_class)
        );
        std::array<std::uint32_t, transaction_word_count> transaction {};
        for (std::size_t index = 0u; index < transaction.size(); ++index) {
          transaction[index] = word(
            base + word_index(record_word_e::transaction_begin) + index
          );
        }
        const auto receipt = authenticate_receipt(
          transaction,
          transaction_token,
          record(record_word_e::expected_work),
          submission_class
        );
        if (record(record_word_e::depth_disposition) !=
            static_cast<std::uint32_t>(receipt.depth)) {
          return false;
        }
        const auto flags = record(record_word_e::flags);
        const auto host_outcome = static_cast<host_subtitle_outcome_e>(
          record(record_word_e::host_subtitle_outcome)
        );
        const auto subtitle_disposition = classify_subtitle_disposition(
          record(record_word_e::expected_work),
          host_outcome,
          receipt,
          flags
        );
        if (record(record_word_e::subtitle_disposition) !=
            static_cast<std::uint32_t>(subtitle_disposition)) {
          return false;
        }
        const bool subtitle_held_with_depth =
          subtitle_disposition == subtitle_disposition_e::held_with_depth;
        if (subtitle_held_with_depth && ordinal != 0u) {
          const auto previous_slot = (oldest_slot + ordinal - 1u) % capacity;
          const auto previous_base = record_base(previous_slot);
          for (std::size_t index = 0u; index < subtitle_locator_word_count; ++index) {
            if (word(base + word_index(record_word_e::subtitle_locator_begin) + index) !=
                word(previous_base +
                     word_index(record_word_e::subtitle_locator_begin) + index)) {
              return false;
            }
          }
          for (std::size_t index = 0u; index < subtitle_condition_word_count; ++index) {
            if (word(base + word_index(record_word_e::subtitle_condition_begin) + index) !=
                word(previous_base +
                     word_index(record_word_e::subtitle_condition_begin) + index)) {
              return false;
            }
          }
        }

        const bool suppressed = (flags & subtitle_suppressed) != 0u;
        if (!suppressed) {
          const auto locator_base =
            base + word_index(record_word_e::subtitle_locator_begin);
          const auto condition_base =
            base + word_index(record_word_e::subtitle_condition_begin);
          const auto locator_flags = word(locator_base + 2u);
          if (word(locator_base) != subtitle_locator_state_schema ||
              word(locator_base + 1u) != subtitle_locator_state_tag ||
              (locator_flags & ~subtitle_locator_known_flags) != 0u ||
              word(locator_base + 4u) > subtitle_locator_rectangle_capacity ||
              word(locator_base + 12u) > subtitle_locator_rectangle_capacity ||
              word(locator_base + 20u) > subtitle_locator_rectangle_capacity ||
              join_u64(word(locator_base + 10u), word(locator_base + 11u)) !=
                analysis_generation ||
              (subtitle_held_with_depth ?
                 (join_u64(word(locator_base + 22u), word(locator_base + 23u)) == 0u ||
                  join_u64(word(locator_base + 22u), word(locator_base + 23u)) >= frame_id) :
                 join_u64(word(locator_base + 22u), word(locator_base + 23u)) != frame_id) ||
              word(locator_base + 27u) != record(record_word_e::field_width) ||
              word(locator_base + 28u) != record(record_word_e::field_height)) {
            return false;
          }
          const auto current_count = word(locator_base + 20u);
          const auto owner_count = word(locator_base + 4u);
          const auto pending_count = word(locator_base + 12u);
          const auto packed_kinds = word(locator_base + subtitle_locator_kind_word);
          const auto owner_kinds =
            (packed_kinds >> subtitle_locator_owner_kind_shift) &
            subtitle_locator_kind_mask;
          const auto pending_kinds =
            (packed_kinds >> subtitle_locator_pending_kind_shift) &
            subtitle_locator_kind_mask;
          const auto current_kinds =
            (packed_kinds >> subtitle_locator_current_kind_shift) &
            subtitle_locator_kind_mask;
          const bool provisional_current =
            (locator_flags & subtitle_locator_flag_provisional_current) != 0u;
          if (provisional_current) {
            const float provisional_target = std::bit_cast<float>(
              word(locator_base + subtitle_locator_provisional_target_word)
            );
            const auto provisional_fade =
              word(locator_base + subtitle_locator_provisional_fade_word);
            const auto trace_rectangle = [locator_base, &word](const std::size_t offset) {
              return std::array<std::uint32_t, 4u> {
                word(locator_base + offset + 0u), word(locator_base + offset + 1u),
                word(locator_base + offset + 2u), word(locator_base + offset + 3u),
              };
            };
            if ((locator_flags & subtitle_locator_flag_pending) == 0u ||
                (locator_flags & subtitle_locator_flag_target_reset) != 0u ||
                pending_count != 1u || current_count != 1u ||
                owner_kinds != 0u || pending_kinds != 0u || current_kinds != 0u ||
                word(locator_base + 21u) != 0u || word(locator_base + 25u) != 0u ||
                !subtitle_target_is_representable(provisional_target)) {
              return false;
            }
            const auto pending_core = trace_rectangle(subtitle_locator_pending_offset);
            const auto current_cover = trace_rectangle(subtitle_locator_current_offset);
            const auto trace_rectangle_is_valid = [&record](
              const std::array<std::uint32_t, 4u> &rectangle
            ) {
              return rectangle[0] < rectangle[2] && rectangle[1] < rectangle[3] &&
                     rectangle[2] <= record(record_word_e::field_width) &&
                     rectangle[3] <= record(record_word_e::field_height);
            };
            if (!trace_rectangle_is_valid(pending_core) ||
                !trace_rectangle_is_valid(current_cover) ||
                current_cover[0] > pending_core[0] ||
                current_cover[1] > pending_core[1] ||
                current_cover[2] < pending_core[2] ||
                current_cover[3] < pending_core[3]) {
              return false;
            }
            if ((locator_flags & subtitle_locator_flag_owner) == 0u ||
                (locator_flags & subtitle_locator_flag_target_valid) == 0u ||
                owner_count != 1u || word(locator_base + 24u) != 2u ||
                (provisional_fade != 1u && provisional_fade != 2u) ||
                !subtitle_provisional_geometry_is_canonical(
                  trace_rectangle(subtitle_locator_owner_offset), pending_core
                )) {
              return false;
            }
          } else if ((locator_flags & subtitle_locator_flag_owner) != 0u &&
                     (word(locator_base + subtitle_locator_provisional_target_word) != 0u ||
                      word(locator_base + subtitle_locator_provisional_fade_word) != 0u)) {
            return false;
          }
          const auto expected_fade_word = provisional_current ?
            subtitle_locator_provisional_fade_word : 24u;
          const auto expected_target_word = provisional_current ?
            subtitle_locator_provisional_target_word : 18u;
          const bool condition_matches_active_state =
            word(condition_base + 0u) == subtitle_condition_param_schema &&
            word(condition_base + 1u) == subtitle_condition_param_tag &&
            word(condition_base + 2u) == current_count &&
            word(condition_base + 3u) == current_kinds &&
            word(condition_base + 4u) == word(locator_base + expected_fade_word) &&
            word(condition_base + 5u) == word(locator_base + expected_target_word);
          bool condition_is_canonical_zero = true;
          for (std::size_t index = 0u; index < subtitle_condition_param_word_count; ++index) {
            condition_is_canonical_zero = condition_is_canonical_zero &&
              word(condition_base + index) == 0u;
          }
          if (current_count != 0u ? !condition_matches_active_state :
                                   !condition_is_canonical_zero) {
            return false;
          }
        }
        if (matched_frame) {
          const auto expected_domain_tag = models::near_identical_input_domain_tag(
            completed.depth_input_region,
            completed.color_space,
            static_cast<std::uint32_t>(completed.model_width),
            static_cast<std::uint32_t>(completed.model_height)
          );
          if (domain_tag != expected_domain_tag || analysis_generation !=
                completed.depth_input_region.analysis_generation ||
              record(record_word_e::source_width) !=
                completed.depth_input_region.width() ||
              record(record_word_e::source_height) !=
                completed.depth_input_region.height() ||
              record(record_word_e::field_width) !=
                static_cast<std::uint32_t>(completed.model_width) ||
              record(record_word_e::field_height) !=
                static_cast<std::uint32_t>(completed.model_height) ||
              ((flags & input_domain_reset) != 0u) != completed.input_domain_reset) {
            return false;
          }
          matched_frame_present = true;
        }
      }
      return matched_frame_present;
    }

    bool subtitle_records_match_completion(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ring,
      const frame &completed,
      const std::uint32_t confirmed_cut_count
    ) {
      using namespace models::host_sbs_gpu_trace;
      if (subtitle_records_match_frame(
            ocr, locator, completed, confirmed_cut_count
          )) {
        if (!completed.gpu_undecided_completion ||
            !gpu_trace_ring_is_canonical(ring, completed)) {
          return true;
        }
        const auto word = [&ring](const std::size_t index) {
          std::uint32_t value = 0u;
          std::memcpy(
            &value, ring.data() + index * sizeof(std::uint32_t), sizeof(value)
          );
          return value;
        };
        const auto next_slot = word(word_index(header_word_e::next_slot));
        const auto count = word(word_index(header_word_e::committed_count));
        const auto oldest_slot = (next_slot + capacity - count) % capacity;
        for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
          const auto base = record_base((oldest_slot + ordinal) % capacity);
          if (join_u64(
                word(base + word_index(record_word_e::frame_low)),
                word(base + word_index(record_word_e::frame_high))
              ) == completed.matched_frame_id) {
            return static_cast<subtitle_disposition_e>(word(
              base + word_index(record_word_e::subtitle_disposition)
            )) != subtitle_disposition_e::held_with_depth;
          }
        }
        return false;
      }
      if (!completed.gpu_undecided_completion ||
          !gpu_trace_ring_is_canonical(ring, completed) ||
          locator.size() != subtitle_locator_state_word_count * sizeof(std::uint32_t)) {
        return false;
      }
      const auto held_frame_id = subtitle_u64(locator, 22u);
      if (held_frame_id == 0u || held_frame_id >= completed.matched_frame_id) {
        return false;
      }
      frame held = completed;
      held.matched_frame_id = held_frame_id;
      held.input_domain_reset = false;
      held.gpu_undecided_completion = false;
      if (!subtitle_records_match_frame(
            ocr, locator, held, confirmed_cut_count
          )) {
        return false;
      }

      const auto word = [&ring](const std::size_t index) {
        std::uint32_t value = 0u;
        std::memcpy(
          &value, ring.data() + index * sizeof(std::uint32_t), sizeof(value)
        );
        return value;
      };
      const auto next_slot = word(word_index(header_word_e::next_slot));
      const auto count = word(word_index(header_word_e::committed_count));
      const auto oldest_slot = (next_slot + capacity - count) % capacity;
      std::optional<std::size_t> matched_locator_base;
      for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        const auto base = record_base((oldest_slot + ordinal) % capacity);
        if (join_u64(
              word(base + word_index(record_word_e::frame_low)),
              word(base + word_index(record_word_e::frame_high))
            ) == completed.matched_frame_id) {
          matched_locator_base =
            base + word_index(record_word_e::subtitle_locator_begin);
        }
      }
      if (!matched_locator_base) {
        return false;
      }
      const auto matched_record_base = *matched_locator_base -
        word_index(record_word_e::subtitle_locator_begin);
      if (static_cast<submission_class_e>(word(
            matched_record_base + word_index(record_word_e::submission_class)
          )) != submission_class_e::gpu_undecided ||
          static_cast<depth_disposition_e>(word(
            matched_record_base + word_index(record_word_e::depth_disposition)
          )) != depth_disposition_e::reuse ||
          static_cast<subtitle_disposition_e>(word(
            matched_record_base + word_index(record_word_e::subtitle_disposition)
          )) != subtitle_disposition_e::held_with_depth) {
        return false;
      }
      for (std::size_t index = 0u; index < subtitle_locator_state_word_count;
           ++index) {
        if (word(*matched_locator_base + index) != subtitle_word(locator, index)) {
          return false;
        }
      }
      return true;
    }

  }  // namespace detail

  namespace {

    struct normalization_state {
      float lower = 0.0f;
      float upper = 0.0f;
      float initialized = 0.0f;
      float frame_state = 0.0f;
    };

    enum class depth_dumpability {
      valid,
      invalid,
      unreadable,
    };

    const char *normalization_frame_state_name(const float value) {
      if (value < 0.5f) {
        return "invalid-held";
      }
      if (value < 1.5f) {
        return "valid-with-history";
      }
      return "first-valid";
    }

    struct texture_snapshot {
      D3D11_TEXTURE2D_DESC desc {};
      std::vector<std::uint8_t> bytes;
      std::size_t row_bytes = 0;
    };

    /**
     * CPU-owned copy of one authenticated render pair. D3D11 resources are read while the
     * immediate context is still owned by the render thread; the background publisher never
     * touches the device or context and therefore cannot race the live pipeline.
     */
    struct captured_dump_job {
      std::filesystem::path root;
      std::filesystem::path trigger;
      std::shared_ptr<std::atomic<bool>> button_request;
      bool by_button = false;
      bool by_file = false;
      frame completed;
      config::video_t::sbs_t cfg;
      const models::depth_coordinate_v2::model_preprocess_contract_t *preprocess = nullptr;

      normalization_state normalization {};
      bool scene_cut_bridge_state_available = false;
      std::vector<std::uint8_t> adaptive_state;

      texture_snapshot source;
      texture_snapshot depth_input_source;
      std::vector<float> model_input;
      std::vector<float> raw_depth;
      texture_snapshot warp_depth;
      texture_snapshot sbs;
      texture_snapshot shadow_coordinate;
      texture_snapshot shadow_candidate;
      texture_snapshot shadow_ownership_refined;
      texture_snapshot shadow_vertical;
      texture_snapshot shadow_vertical_conditioned;
      texture_snapshot shadow_base_final;
      texture_snapshot shadow_final;
      std::vector<std::uint8_t> subtitle_ocr_record;
      std::vector<std::uint8_t> subtitle_locator_state;
      std::vector<std::uint8_t> gpu_trace_ring;
      std::vector<float> shadow_state;
      std::vector<float> shadow_frame_stats;
      bool warp_map_available = false;
      texture_snapshot warp_map;
      bool warp_mask_available = false;
      texture_snapshot warp_mask;
    };

    struct staged_buffer {
      Microsoft::WRL::ComPtr<ID3D11Buffer> resource;
      std::size_t byte_count = 0;
    };

    struct staged_texture {
      Microsoft::WRL::ComPtr<ID3D11Texture2D> resource;
      D3D11_TEXTURE2D_DESC desc {};
      std::size_t row_bytes = 0;
    };

  }  // namespace

  namespace detail {

    namespace {
      float normalize_scalar_preview_value(
        const float value,
        const float low,
        const float high,
        const bool midpoint_when_collapsed
      ) noexcept {
        const float span = high - low;
        const float scale = std::max(1.0f, std::max(std::fabs(low), std::fabs(high)));
        if (!(span > std::numeric_limits<float>::epsilon() * scale)) {
          return midpoint_when_collapsed ? 0.5f : std::clamp(value, 0.0f, 1.0f);
        }
        return std::clamp((value - low) / span, 0.0f, 1.0f);
      }
    }  // namespace

    std::size_t bounded_collection_chunk_bytes(
      const std::size_t remaining_bytes,
      const std::size_t alignment,
      const std::size_t available_budget,
      const bool poll_is_empty
    ) noexcept {
      if (remaining_bytes == 0 || alignment == 0 ||
          remaining_bytes % alignment != 0) {
        return 0;
      }
      if (available_budget >= alignment) {
        return std::min(
          remaining_bytes,
          available_budget - available_budget % alignment
        );
      }
      return poll_is_empty ? std::min(remaining_bytes, alignment) : 0;
    }

    enum class collection_stage_e : std::uint8_t {
      shadow_state,
      shadow_frame_stats,
      subtitle_ocr_record,
      subtitle_locator_state,
      gpu_trace_ring,
      validate_gpu_trace,
      validate_evidence,
      scene_normalization,
      scene_adaptive,
      scene_decode,
      model_input,
      raw_depth,
      warp_depth,
      shadow_coordinate,
      shadow_candidate,
      shadow_ownership_refined,
      shadow_vertical,
      shadow_vertical_conditioned,
      shadow_base_final,
      shadow_final,
      source,
      depth_input_source,
      sbs,
      warp_map,
      warp_mask,
      complete,
    };

    /**
     * One exact-frame D3D11 staging batch. Every copy and the terminal event query are submitted
     * together on the owning immediate context before any live texture can be reused.
     */
    struct pending_gpu_capture {
      captured_dump_job job;
      Microsoft::WRL::ComPtr<ID3D11Query> completion;

      staged_texture source;
      staged_texture depth_input_source;
      staged_buffer model_input;
      staged_buffer raw_depth;
      staged_texture warp_depth;
      staged_texture sbs;
      staged_texture shadow_coordinate;
      staged_texture shadow_candidate;
      staged_texture shadow_ownership_refined;
      staged_texture shadow_vertical;
      staged_texture shadow_vertical_conditioned;
      staged_texture shadow_base_final;
      staged_texture shadow_final;
      staged_buffer subtitle_ocr_record;
      staged_buffer subtitle_locator_state;
      staged_buffer gpu_trace_ring;
      staged_buffer shadow_state;
      staged_buffer shadow_frame_stats;
      staged_texture warp_map;
      staged_texture warp_mask;
      staged_buffer depth_frame_state;
      staged_buffer adaptive_state;

      std::vector<std::uint8_t> normalization_state_bytes;
      collection_stage_e collection_stage = collection_stage_e::shadow_state;

      bool subtitle_slr13_active = false;
      bool gpu_trace_requested = false;
      bool depth_input_source_available = false;
      bool scene_cut_bridge_requested = false;
      bool gpu_ready = false;
      std::uint64_t retry_token = 0;
      std::chrono::steady_clock::time_point submitted_at {};
      std::chrono::steady_clock::time_point collection_started_at {};
      double gpu_ready_age_ms = 0.0;
      double cpu_collection_ms = 0.0;
      std::uint32_t collection_poll_count = 0;
    };

  }  // namespace detail

  namespace {

    struct dump_publish_result {
      bool success = false;
      bool trigger_remove_failed = false;
      std::filesystem::path published_path;
      std::string error;
    };

    dump_publish_result publish_captured_dump(const captured_dump_job &job);

    struct scalar_stats {
      std::size_t finite_count = 0;
      float minimum = std::numeric_limits<float>::quiet_NaN();
      float maximum = std::numeric_limits<float>::quiet_NaN();
      float preview_low = std::numeric_limits<float>::quiet_NaN();
      float preview_high = std::numeric_limits<float>::quiet_NaN();
    };

    struct raw_depth_dump_stats: scalar_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
    };

    struct warp_map_dump_stats: scalar_stats {
      std::uint32_t width = 0;
      std::uint32_t height = 0;
      std::uint32_t eye_width = 0;
      std::uint32_t eye_height = 0;
      float content_scale_x = 1.0f;
      float content_scale_y = 1.0f;
      float displacement_preview_abs_px = 0.0f;
    };

    std::uint32_t format_bytes_per_pixel(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return 8;
        default:
          return 0;
      }
    }

    std::string format_name(const DXGI_FORMAT format) {
      switch (format) {
        case DXGI_FORMAT_R32_FLOAT:
          return "DXGI_FORMAT_R32_FLOAT";
        case DXGI_FORMAT_R32_UINT:
          return "DXGI_FORMAT_R32_UINT";
        case DXGI_FORMAT_B8G8R8A8_UNORM:
          return "DXGI_FORMAT_B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8X8_UNORM:
          return "DXGI_FORMAT_B8G8R8X8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
          return "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UNORM:
          return "DXGI_FORMAT_R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
          return "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
          return "DXGI_FORMAT_R16G16B16A16_FLOAT";
        default:
          return "DXGI_FORMAT_UNKNOWN_" + std::to_string(static_cast<unsigned>(format));
      }
    }

    nlohmann::json texture_description(const texture_snapshot &snapshot) {
      return {
        {"width", snapshot.desc.Width},
        {"height", snapshot.desc.Height},
        {"format", format_name(snapshot.desc.Format)},
        {"format_value", static_cast<unsigned>(snapshot.desc.Format)},
      };
    }

    inline float half_to_float(const std::uint16_t h) {
      const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16u;
      int exp = (h & 0x7C00u) >> 10u;
      std::uint32_t mant = h & 0x03FFu;
      std::uint32_t bits;
      if (exp == 0) {
        if (mant == 0) {
          bits = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x0400u)) {
            mant <<= 1u;
            --exp;
          }
          mant &= 0x03FFu;
          bits = sign | (static_cast<std::uint32_t>(exp) << 23u) | (mant << 13u);
        }
      } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13u);
      } else {
        bits = sign |
               (static_cast<std::uint32_t>(exp - 15 + 127) << 23u) |
               (mant << 13u);
      }
      return std::bit_cast<float>(bits);
    }

    inline std::uint8_t encode_srgb(float value) {
      value = std::clamp(value, 0.0f, 1.0f);
      const float encoded = value <= 0.0031308f ?
                              12.92f * value :
                              1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
      return static_cast<std::uint8_t>(std::lround(encoded * 255.0f));
    }

    inline std::uint8_t encode_unit(float value) {
      return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)
      );
    }

    inline void tonemap_scrgb(float &r, float &g, float &b) {
      r = std::max(r, 0.0f);
      g = std::max(g, 0.0f);
      b = std::max(b, 0.0f);
      const float luminance =
        std::max(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.0f);
      const float tone_scale = 1.0f / (1.0f + luminance);
      r *= tone_scale;
      g *= tone_scale;
      b *= tone_scale;
      const float gamut_scale =
        1.0f / std::max(1.0f, std::max(r, std::max(g, b)));
      r *= gamut_scale;
      g *= gamut_scale;
      b *= gamut_scale;
    }

    inline void colormap_jet(float value, std::uint8_t &r, std::uint8_t &g, std::uint8_t &b) {
      const float t = std::clamp(value, 0.0f, 1.0f);
      const auto channel = [](const float x) {
        return static_cast<std::uint8_t>(
          std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)
        );
      };
      r = channel(1.5f - std::fabs(4.0f * t - 3.0f));
      g = channel(1.5f - std::fabs(4.0f * t - 2.0f));
      b = channel(1.5f - std::fabs(4.0f * t - 1.0f));
    }

    bool write_bytes(
      const std::filesystem::path &path,
      const void *data,
      const std::size_t size
    ) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      if (size != 0) {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
      }
      out.flush();
      return out.good();
    }

    bool write_text(const std::filesystem::path &path, const std::string &text) {
      return write_bytes(path, text.data(), text.size());
    }

    bool write_json(const std::filesystem::path &path, const nlohmann::json &value) {
      try {
        return write_text(path, value.dump(2) + "\n");
      } catch (const std::exception &error) {
        BOOST_LOG(warning) << "SBS debug dump: JSON serialization failed for "sv
                           << path.string() << ": " << error.what();
        return false;
      }
    }

    bool write_png(
      const std::filesystem::path &path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<std::uint8_t> &rgb
    ) {
      if (
        width == 0 || height == 0 ||
        static_cast<std::uint64_t>(width) * height >
          std::numeric_limits<std::size_t>::max() / 3u ||
        rgb.size() != static_cast<std::size_t>(width) * height * 3u
      ) {
        return false;
      }

      const std::size_t scanline = 1u + static_cast<std::size_t>(width) * 3u;
      if (static_cast<std::size_t>(height) > SIZE_MAX / scanline) {
        return false;
      }
      std::vector<std::uint8_t> raw(static_cast<std::size_t>(height) * scanline);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::size_t output_offset = static_cast<std::size_t>(y) * scanline;
        raw[output_offset] = 0;
        std::memcpy(
          raw.data() + output_offset + 1u,
          rgb.data() + static_cast<std::size_t>(y) * width * 3u,
          static_cast<std::size_t>(width) * 3u
        );
      }

      if (
        raw.size() > std::numeric_limits<uLong>::max() ||
        raw.size() > std::numeric_limits<uLongf>::max()
      ) {
        return false;
      }
      uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
      std::vector<std::uint8_t> compressed(compressed_size);
      if (
        compress2(
          compressed.data(),
          &compressed_size,
          raw.data(),
          static_cast<uLong>(raw.size()),
          Z_BEST_SPEED
        ) != Z_OK
      ) {
        return false;
      }

      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        return false;
      }
      const auto write_be32 = [](const std::uint32_t value, std::uint8_t *bytes) {
        bytes[0] = static_cast<std::uint8_t>(value >> 24u);
        bytes[1] = static_cast<std::uint8_t>(value >> 16u);
        bytes[2] = static_cast<std::uint8_t>(value >> 8u);
        bytes[3] = static_cast<std::uint8_t>(value);
      };
      const auto write_chunk = [&](const char *type, const std::uint8_t *data, const std::uint32_t size) {
        std::uint8_t encoded_size[4];
        write_be32(size, encoded_size);
        out.write(reinterpret_cast<const char *>(encoded_size), sizeof(encoded_size));
        out.write(type, 4);
        if (size != 0) {
          out.write(reinterpret_cast<const char *>(data), size);
        }
        uLong crc = crc32(0, reinterpret_cast<const Bytef *>(type), 4);
        if (size != 0) {
          crc = crc32(crc, data, size);
        }
        std::uint8_t encoded_crc[4];
        write_be32(static_cast<std::uint32_t>(crc), encoded_crc);
        out.write(reinterpret_cast<const char *>(encoded_crc), sizeof(encoded_crc));
        return out.good();
      };

      static constexpr std::uint8_t signature[8] {
        0x89,
        'P',
        'N',
        'G',
        0x0D,
        0x0A,
        0x1A,
        0x0A
      };
      out.write(reinterpret_cast<const char *>(signature), sizeof(signature));
      std::uint8_t header[13] {};
      write_be32(width, header);
      write_be32(height, header + 4);
      header[8] = 8;
      header[9] = 2;
      const bool ok =
        out.good() &&
        write_chunk("IHDR", header, sizeof(header)) &&
        write_chunk(
          "IDAT",
          compressed.data(),
          static_cast<std::uint32_t>(compressed_size)
        ) &&
        write_chunk("IEND", nullptr, 0);
      out.flush();
      return ok && out.good();
    }

    enum class collect_status_e {
      ready,
      partial,
      not_ready,
      failed,
    };

    class collection_budget {
    public:
      explicit collection_budget(const std::chrono::steady_clock::time_point started) noexcept:
          deadline_(started + std::chrono::milliseconds(2)) {
      }

      std::size_t next_chunk(
        const std::size_t remaining_bytes,
        const std::size_t alignment
      ) const noexcept {
        if (remaining_maps_ == 0 ||
            std::chrono::steady_clock::now() >= deadline_) {
          return 0;
        }
        return detail::bounded_collection_chunk_bytes(
          remaining_bytes,
          alignment,
          remaining_bytes_,
          copied_bytes_ == 0
        );
      }

      void record_copy(const std::size_t bytes) noexcept {
        copied_bytes_ += bytes;
        remaining_bytes_ = bytes >= remaining_bytes_ ? 0 : remaining_bytes_ - bytes;
        if (remaining_maps_ != 0) {
          --remaining_maps_;
        }
      }

      bool exhausted() const noexcept {
        return remaining_bytes_ == 0 || remaining_maps_ == 0 ||
               std::chrono::steady_clock::now() >= deadline_;
      }

    private:
      std::chrono::steady_clock::time_point deadline_ {};
      std::size_t remaining_bytes_ = detail::cpu_collection_byte_budget;
      std::size_t copied_bytes_ = 0;
      unsigned remaining_maps_ = 8;
    };

    bool stage_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      const std::size_t required_bytes,
      staged_buffer &staging
    ) {
      if (!device || !ctx || !srv || required_bytes == 0) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
      if (!resource || FAILED(resource.As(&buffer))) {
        return false;
      }

      D3D11_BUFFER_DESC source_desc {};
      buffer->GetDesc(&source_desc);
      if (required_bytes > source_desc.ByteWidth) {
        return false;
      }
      D3D11_BUFFER_DESC staging_desc = source_desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      staging_desc.StructureByteStride = 0;
      if (FAILED(device->CreateBuffer(&staging_desc, nullptr, &staging.resource))) {
        return false;
      }
      staging.byte_count = required_bytes;
      ctx->CopyResource(staging.resource.Get(), buffer.Get());
      return true;
    }

    bool stage_texture(
      ID3D11Device *device,
      ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *srv,
      staged_texture &staging
    ) {
      if (!device || !ctx || !srv) {
        return false;
      }
      Microsoft::WRL::ComPtr<ID3D11Resource> resource;
      srv->GetResource(&resource);
      Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
      if (!resource || FAILED(resource.As(&texture))) {
        return false;
      }
      texture->GetDesc(&staging.desc);
      const std::uint32_t bytes_per_pixel = format_bytes_per_pixel(staging.desc.Format);
      if (
        bytes_per_pixel == 0 || staging.desc.Width == 0 || staging.desc.Height == 0 ||
        staging.desc.ArraySize != 1 || staging.desc.MipLevels != 1 ||
        staging.desc.SampleDesc.Count != 1
      ) {
        return false;
      }
      staging.row_bytes =
        static_cast<std::size_t>(staging.desc.Width) * bytes_per_pixel;
      if (
        static_cast<std::size_t>(staging.desc.Height) >
        SIZE_MAX / staging.row_bytes
      ) {
        return false;
      }

      D3D11_TEXTURE2D_DESC staging_desc = staging.desc;
      staging_desc.Usage = D3D11_USAGE_STAGING;
      staging_desc.BindFlags = 0;
      staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      staging_desc.MiscFlags = 0;
      if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging.resource))) {
        return false;
      }
      ctx->CopyResource(staging.resource.Get(), texture.Get());
      return true;
    }

    collect_status_e collect_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      std::vector<std::uint8_t> &bytes,
      collection_budget &budget
    ) {
      if (!ctx || !staging.resource || staging.byte_count == 0) {
        return collect_status_e::failed;
      }
      if (bytes.size() > staging.byte_count) {
        return collect_status_e::failed;
      }
      if (bytes.size() == staging.byte_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      bytes.reserve(staging.byte_count);
      const std::size_t copy_bytes = budget.next_chunk(
        staging.byte_count - bytes.size(), 1u
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData) {
        return collect_status_e::failed;
      }
      try {
        const auto *source = static_cast<const std::uint8_t *>(mapped.pData) + bytes.size();
        bytes.insert(bytes.end(), source, source + copy_bytes);
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (bytes.size() == staging.byte_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

    collect_status_e collect_float_buffer(
      ID3D11DeviceContext *ctx,
      staged_buffer &staging,
      const std::size_t value_count,
      std::vector<float> &values,
      collection_budget &budget
    ) {
      if (value_count > SIZE_MAX / sizeof(float) ||
          staging.byte_count != value_count * sizeof(float)) {
        return collect_status_e::failed;
      }
      if (!ctx || !staging.resource || values.size() > value_count) {
        return collect_status_e::failed;
      }
      if (values.size() == value_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      values.reserve(value_count);
      const std::size_t copy_bytes = budget.next_chunk(
        (value_count - values.size()) * sizeof(float), sizeof(float)
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData) {
        return collect_status_e::failed;
      }
      try {
        const auto *source = static_cast<const float *>(mapped.pData) + values.size();
        values.insert(values.end(), source, source + copy_bytes / sizeof(float));
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (values.size() == value_count) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

    collect_status_e collect_texture(
      ID3D11DeviceContext *ctx,
      staged_texture &staging,
      texture_snapshot &snapshot,
      collection_budget &budget
    ) {
      if (!ctx || !staging.resource || staging.row_bytes == 0) {
        return collect_status_e::failed;
      }
      const std::size_t total_bytes =
        static_cast<std::size_t>(staging.desc.Height) * staging.row_bytes;
      if (snapshot.bytes.size() > total_bytes ||
          snapshot.bytes.size() % staging.row_bytes != 0) {
        return collect_status_e::failed;
      }
      snapshot.desc = staging.desc;
      snapshot.row_bytes = staging.row_bytes;
      if (snapshot.bytes.size() == total_bytes) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      snapshot.bytes.reserve(total_bytes);
      const std::size_t copy_bytes = budget.next_chunk(
        total_bytes - snapshot.bytes.size(), staging.row_bytes
      );
      if (copy_bytes == 0) {
        return collect_status_e::partial;
      }
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT status = ctx->Map(
        staging.resource.Get(),
        0,
        D3D11_MAP_READ,
        D3D11_MAP_FLAG_DO_NOT_WAIT,
        &mapped
      );
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return collect_status_e::not_ready;
      }
      if (FAILED(status) || !mapped.pData || mapped.RowPitch < staging.row_bytes) {
        if (SUCCEEDED(status)) {
          ctx->Unmap(staging.resource.Get(), 0);
        }
        return collect_status_e::failed;
      }

      const std::size_t first_row = snapshot.bytes.size() / snapshot.row_bytes;
      const std::size_t row_count = copy_bytes / snapshot.row_bytes;
      try {
        const auto *source = static_cast<const std::uint8_t *>(mapped.pData) +
                             first_row * mapped.RowPitch;
        if (mapped.RowPitch == snapshot.row_bytes) {
          snapshot.bytes.insert(snapshot.bytes.end(), source, source + copy_bytes);
        } else {
          for (std::size_t y = 0; y < row_count; ++y) {
            const auto *row = source + y * mapped.RowPitch;
            snapshot.bytes.insert(
              snapshot.bytes.end(), row, row + snapshot.row_bytes
            );
          }
        }
      } catch (...) {
        ctx->Unmap(staging.resource.Get(), 0);
        throw;
      }
      ctx->Unmap(staging.resource.Get(), 0);
      budget.record_copy(copy_bytes);
      if (snapshot.bytes.size() == total_bytes) {
        staging.resource.Reset();
        return collect_status_e::ready;
      }
      return collect_status_e::partial;
    }

    bool texture_to_rgb(
      const texture_snapshot &snapshot,
      const models::input_color_space color_space,
      std::vector<std::uint8_t> &rgb
    ) {
      const auto width = snapshot.desc.Width;
      const auto height = snapshot.desc.Height;
      if (
        static_cast<std::uint64_t>(width) * height >
        SIZE_MAX / 3u
      ) {
        return false;
      }
      rgb.resize(static_cast<std::size_t>(width) * height * 3u);
      for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t *input =
          snapshot.bytes.data() + static_cast<std::size_t>(y) * snapshot.row_bytes;
        std::uint8_t *output =
          rgb.data() + static_cast<std::size_t>(y) * width * 3u;
        for (std::uint32_t x = 0; x < width; ++x) {
          std::uint8_t r;
          std::uint8_t g;
          std::uint8_t b;
          switch (snapshot.desc.Format) {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
              {
                const auto *pixel =
                  reinterpret_cast<const std::uint16_t *>(input + static_cast<std::size_t>(x) * 8u);
                float rf = half_to_float(pixel[0]);
                float gf = half_to_float(pixel[1]);
                float bf = half_to_float(pixel[2]);
                if (!std::isfinite(rf) || !std::isfinite(gf) || !std::isfinite(bf)) {
                  r = 255;
                  g = 0;
                  b = 255;
                } else {
                  if (color_space == models::input_color_space::scrgb_hdr) {
                    tonemap_scrgb(rf, gf, bf);
                  }
                  if (color_space == models::input_color_space::srgb) {
                    // Some capture paths conservatively retain FP16 storage for sRGB code
                    // values. The warp copies those values unchanged, so applying an OETF here
                    // would double-gamma both source.png and sbs.png.
                    r = encode_unit(rf);
                    g = encode_unit(gf);
                    b = encode_unit(bf);
                  } else {
                    r = encode_srgb(rf);
                    g = encode_srgb(gf);
                    b = encode_srgb(bf);
                  }
                }
                break;
              }
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8X8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                b = pixel[0];
                g = pixel[1];
                r = pixel[2];
                break;
              }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
              {
                const std::uint8_t *pixel = input + static_cast<std::size_t>(x) * 4u;
                r = pixel[0];
                g = pixel[1];
                b = pixel[2];
                break;
              }
            default:
              return false;
          }
          output[static_cast<std::size_t>(x) * 3u + 0u] = r;
          output[static_cast<std::size_t>(x) * 3u + 1u] = g;
          output[static_cast<std::size_t>(x) * 3u + 2u] = b;
        }
      }
      return true;
    }

    bool write_color_preview(
      const std::filesystem::path &path,
      const texture_snapshot &snapshot,
      const models::input_color_space color_space
    ) {
      std::vector<std::uint8_t> rgb;
      return texture_to_rgb(snapshot, color_space, rgb) &&
             write_png(path, snapshot.desc.Width, snapshot.desc.Height, rgb);
    }

    bool texture_float_values(
      const texture_snapshot &snapshot,
      std::vector<float> &values
    ) {
      if (snapshot.desc.Format != DXGI_FORMAT_R32_FLOAT) {
        return false;
      }
      const std::size_t count =
        static_cast<std::size_t>(snapshot.desc.Width) * snapshot.desc.Height;
      if (snapshot.bytes.size() != count * sizeof(float)) {
        return false;
      }
      values.resize(count);
      std::memcpy(values.data(), snapshot.bytes.data(), snapshot.bytes.size());
      return true;
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values);

    struct scalar_preview_normalization_t {
      float low;
      float high;
      bool midpoint_when_collapsed;
    };

    bool write_normalized_scalar_previews(
      const std::filesystem::path &gray_path,
      const std::filesystem::path &heat_path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<float> &values,
      const scalar_preview_normalization_t normalization
    ) {
      std::vector<std::uint8_t> gray(values.size() * 3u);
      std::vector<std::uint8_t> heat(values.size() * 3u);
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
          gray[index * 3u + 0u] = heat[index * 3u + 0u] = 255;
          gray[index * 3u + 1u] = heat[index * 3u + 1u] = 0;
          gray[index * 3u + 2u] = heat[index * 3u + 2u] = 255;
          continue;
        }
        const float normalized = detail::normalize_scalar_preview_value(
          values[index],
          normalization.low,
          normalization.high,
          normalization.midpoint_when_collapsed
        );
        const std::uint8_t encoded = encode_unit(normalized);
        gray[index * 3u + 0u] = encoded;
        gray[index * 3u + 1u] = encoded;
        gray[index * 3u + 2u] = encoded;
        colormap_jet(
          normalized,
          heat[index * 3u + 0u],
          heat[index * 3u + 1u],
          heat[index * 3u + 2u]
        );
      }
      return write_png(gray_path, width, height, gray) &&
             write_png(heat_path, width, height, heat);
    }

    bool write_scalar_previews(
      const std::filesystem::path &gray_path,
      const std::filesystem::path &heat_path,
      const texture_snapshot &snapshot
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      return write_normalized_scalar_previews(
        gray_path,
        heat_path,
        snapshot.desc.Width,
        snapshot.desc.Height,
        values,
        {0.0f, 1.0f, false}
      );
    }

    bool write_float_texture_artifacts(
      const std::filesystem::path &data_path,
      const std::filesystem::path &shape_path,
      const texture_snapshot &snapshot,
      const std::string_view stage
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      const scalar_stats stats = calculate_scalar_stats(values);
      if (
        !write_bytes(
          data_path,
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }
      nlohmann::json shape {
        {"schema", 1},
        {"width", snapshot.desc.Width},
        {"height", snapshot.desc.Height},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"stage", std::string(stage)},
        {"finite_count", stats.finite_count},
        {"sample_count", values.size()},
      };
      if (stats.finite_count != 0) {
        shape["minimum"] = stats.minimum;
        shape["maximum"] = stats.maximum;
      }
      return write_json(shape_path, shape);
    }

    scalar_stats calculate_scalar_stats(const std::vector<float> &values) {
      scalar_stats stats;
      std::vector<float> finite;
      finite.reserve(values.size());
      for (const float value : values) {
        if (std::isfinite(value)) {
          finite.push_back(value);
        }
      }
      stats.finite_count = finite.size();
      if (finite.empty()) {
        return stats;
      }
      std::sort(finite.begin(), finite.end());
      const auto percentile = [&](const double fraction) {
        const std::size_t index = static_cast<std::size_t>(
          std::lround(fraction * static_cast<double>(finite.size() - 1u))
        );
        return finite[std::min(index, finite.size() - 1u)];
      };
      stats.minimum = finite.front();
      stats.maximum = finite.back();
      stats.preview_low = percentile(0.02);
      stats.preview_high = percentile(0.98);
      const float scale = std::max(
        1.0f,
        std::max(std::fabs(stats.preview_low), std::fabs(stats.preview_high))
      );
      if (
        !(stats.preview_high - stats.preview_low >
          std::numeric_limits<float>::epsilon() * scale)
      ) {
        stats.preview_low = stats.minimum;
        stats.preview_high = stats.maximum;
      }
      return stats;
    }

    bool write_percentile_previews(
      const std::filesystem::path &gray_path,
      const std::filesystem::path &heat_path,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::vector<float> &values,
      const scalar_stats &stats
    ) {
      if (stats.finite_count == 0) {
        return false;
      }
      return write_normalized_scalar_previews(
        gray_path,
        heat_path,
        width,
        height,
        values,
        {stats.preview_low, stats.preview_high, true}
      );
    }

    bool dump_model_input(
      const std::vector<float> &values,
      const int width,
      const int height,
      const std::filesystem::path &dir,
      const models::depth_coordinate_v2::model_preprocess_contract_t &preprocess
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (pixel_count > SIZE_MAX / (3u * sizeof(float))) {
        return false;
      }
      if (
        values.size() != static_cast<std::size_t>(pixel_count) * 3u ||
        !write_bytes(
          dir / "model_input.f32",
          values.data(),
          values.size() * sizeof(float)
        )
      ) {
        return false;
      }

      std::vector<std::uint8_t> rgb(static_cast<std::size_t>(pixel_count) * 3u);
      const std::size_t plane_size = static_cast<std::size_t>(pixel_count);
      for (std::size_t pixel = 0; pixel < plane_size; ++pixel) {
        for (std::size_t channel = 0; channel < 3u; ++channel) {
          const float normalized = values[channel * plane_size + pixel];
          if (!std::isfinite(normalized)) {
            rgb[pixel * 3u + 0u] = 255;
            rgb[pixel * 3u + 1u] = 0;
            rgb[pixel * 3u + 2u] = 255;
            break;
          }
          // rgb_to_nchw_cs stores already-sRGB model values after ImageNet normalization.
          // Reverse only mean/std here; applying the OETF again would corrupt the preview.
          rgb[pixel * 3u + channel] =
            encode_unit(
              normalized * preprocess.imagenet_std[channel] +
              preprocess.imagenet_mean[channel]
            );
        }
      }
      const nlohmann::json shape {
        {"schema", preprocess.model_input_schema},
        {"width", width},
        {"height", height},
        {"dtype", std::string {preprocess.dtype}},
        {"layout", std::string {preprocess.layout}},
        {"channels", {
                       std::string {preprocess.channels[0]},
                       std::string {preprocess.channels[1]},
                       std::string {preprocess.channels[2]},
                     }},
        {"stage", std::string {preprocess.stage}},
        {"imagenet_mean", preprocess.imagenet_mean},
        {"imagenet_std", preprocess.imagenet_std},
        {"preview", {
                      {"file", "model_input.png"},
                      {"operation", "channel * std + mean, clamped to [0,1]"},
                      {"extra_srgb_oetf", false},
                      {"nonfinite_color", "magenta"},
                    }},
      };
      return write_png(
               dir / "model_input.png",
               static_cast<std::uint32_t>(width),
               static_cast<std::uint32_t>(height),
               rgb
             ) &&
             write_json(dir / "model_input_shape.json", shape);
    }

    bool dump_raw_depth(
      const std::vector<float> &values,
      const int width,
      const int height,
      const std::filesystem::path &dir,
      raw_depth_dump_stats &stats
    ) {
      if (width <= 0 || height <= 0) {
        return false;
      }
      const std::uint64_t value_count =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (value_count > SIZE_MAX / sizeof(float)) {
        return false;
      }
      if (values.size() != static_cast<std::size_t>(value_count)) {
        return false;
      }
      const scalar_stats scalar = calculate_scalar_stats(values);
      static_cast<scalar_stats &>(stats) = scalar;
      stats.width = static_cast<std::uint32_t>(width);
      stats.height = static_cast<std::uint32_t>(height);
      if (
        scalar.finite_count == 0 ||
        !write_bytes(
          dir / "raw_depth.f32",
          values.data(),
          values.size() * sizeof(float)
        ) ||
        !write_percentile_previews(
          dir / "raw_depth.png",
          dir / "raw_depth_heat.png",
          stats.width,
          stats.height,
          values,
          stats
        )
      ) {
        return false;
      }
      const nlohmann::json shape {
        {"schema", 1},
        {"width", stats.width},
        {"height", stats.height},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"stage", "raw model output before transform, robust normalization, temporal EMA, or curvature"},
        {"finite_count", stats.finite_count},
        {"sample_count", values.size()},
        {"minimum", stats.minimum},
        {"maximum", stats.maximum},
        {"preview_normalization", "finite p2-p98"},
        {"preview_low", stats.preview_low},
        {"preview_high", stats.preview_high},
        {"nonfinite_color", "magenta"},
      };
      return write_json(dir / "raw_shape.json", shape);
    }

    bool dump_shadow_float_texture(
      const texture_snapshot &snapshot,
      const std::filesystem::path &dir,
      const std::string_view stem,
      const std::string_view stage
    ) {
      std::vector<float> values;
      if (!texture_float_values(snapshot, values)) {
        return false;
      }
      const scalar_stats stats = calculate_scalar_stats(values);
      if (stats.finite_count != values.size()) {
        return false;
      }
      const std::string name(stem);
      return write_float_texture_artifacts(
               dir / (name + ".f32"),
               dir / (name + "_shape.json"),
               snapshot,
               stage
             ) &&
             write_percentile_previews(
               dir / (name + ".png"),
               dir / (name + "_heat.png"),
               snapshot.desc.Width,
               snapshot.desc.Height,
               values,
               stats
             );
    }

    bool dump_parallax_v2_state(
      const frame &completed,
      const std::vector<float> &state,
      const std::vector<float> &frame_stats,
      const std::filesystem::path &dir,
      nlohmann::json &summary
    ) {
      using namespace models::depth_coordinate_v2;
      if (!parallax_v2_shader_identity_matches_contract(
            completed.parallax_v2_shader_provenance
          )) {
        return false;
      }
      const auto &shader_identity = *completed.parallax_v2_shader_provenance;
      if (state.size() != state_float_count ||
          frame_stats.size() != frame_stats_float_count) {
        return false;
      }
      bool state_finite = true;
      for (std::size_t index = 0; index < state.size(); ++index) {
        if (state_fields[index].gpu_encoding == state_gpu_encoding_e::float_value &&
            !std::isfinite(state[index])) {
          state_finite = false;
          break;
        }
      }
      if (!state_finite || !std::all_of(frame_stats.begin(), frame_stats.end(), [](float value) {
            return std::isfinite(value);
          })) {
        return false;
      }
      state_words_t state_words {};
      std::memcpy(state_words.data(), state.data(), sizeof(state_words));
      const bool runtime_constants_valid = parallax_runtime_constants_are_valid(
        completed.parallax_v2_raw_coordinate_scale,
        completed.parallax_v2_requested_pop_strength,
        completed.parallax_v2_requested_gain
      );
      const bool state_semantics_valid = parallax_state_words_are_authenticated(
        state_words,
        completed.parallax_v2_raw_coordinate_scale
      );
      const bool state_frame_valid = state[frame_valid] > 0.5f;
      const auto calibration_revision_value = state_words[calibration_revision];
      const auto camera_center_integrity_value = state_words[camera_center_integrity_bits];
      const auto renderer_authorization_value = state_words[renderer_authorization_bits];
      const bool camera_initialized =
        state[inverse_scale] > 0.0f &&
        acquired_calibration_revision_is_valid(calibration_revision_value);
      const float valid_count = frame_stats[frame_stat_valid_count];
      const float texel_count = frame_stats[frame_stat_texel_count];
      const bool frame_is_valid = frame_stats[frame_stat_valid] > 0.5f;
      const bool expected_state_frame_valid =
        frame_is_valid &&
        frame_stats[frame_stat_population_std] > collapse_abs_epsilon;
      const bool frame_counts_valid =
        valid_count >= 0.0f && texel_count > 0.0f && valid_count <= texel_count &&
        valid_count == std::floor(valid_count) && texel_count == std::floor(texel_count) &&
        texel_count <= static_cast<float>(std::numeric_limits<std::uint32_t>::max());
      const bool frame_semantics_valid =
        (frame_stats[frame_stat_valid] == 0.0f ||
         frame_stats[frame_stat_valid] == 1.0f) &&
        frame_counts_valid && frame_stats[frame_stat_population_std] >= 0.0f &&
        (frame_is_valid ?
           (valid_count == texel_count &&
            frame_stats[frame_stat_maximum] >= frame_stats[frame_stat_minimum]) :
           (frame_stats[frame_stat_mean] == 0.0f &&
            frame_stats[frame_stat_population_std] == 0.0f &&
            frame_stats[frame_stat_minimum] == 0.0f &&
            frame_stats[frame_stat_maximum] == 0.0f));
      if (!runtime_constants_valid || !state_semantics_valid ||
          !frame_semantics_valid || state_frame_valid != expected_state_frame_valid) {
        return false;
      }

      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      for (std::size_t index = 0; index < state.size(); ++index) {
        const auto &descriptor = state_fields[index];
        const std::string name {descriptor.name};
        if (descriptor.gpu_encoding == state_gpu_encoding_e::uint_bits) {
          const auto bits = std::bit_cast<std::uint32_t>(state[index]);
          fields.push_back({
            {"index", index},
            {"name", name},
            {"type", "uint32-bitcast"},
            {"value", bits},
          });
          named_values[name] = bits;
          continue;
        }
        fields.push_back({
          {"index", index},
          {"name", name},
          {"type", "float32"},
          {"value", state[index]},
        });
        named_values[name] = state[index];
      }

      const float effective_gain_value = state_frame_valid ?
        completed.parallax_v2_requested_gain : 0.0f;
      const float latched_scale_value = camera_initialized ?
        1.0f / state[inverse_scale] : 0.0f;
      const auto confirmed_cut_count_value = state_words[confirmed_cut_count];
      const nlohmann::json decoded {
        {"frame_valid", state_frame_valid},
        {"camera_valid", camera_initialized},
        {"calibration_revision", calibration_revision_value},
        {"confirmed_cut_count", confirmed_cut_count_value},
        {"contract_tag", contract_tag},
        {"requested_gain", completed.parallax_v2_requested_gain},
        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
        {"latched_scale", latched_scale_value},
        {"convergence_curve", state[convergence_curve]},
        {"container_scale", state[container_scale]},
        {"effective_gain", effective_gain_value},
        {"camera_center_integrity_bits", camera_center_integrity_value},
        {"renderer_authorization_bits", renderer_authorization_value},
      };
      const nlohmann::json state_json {
        {"schema", shadow_state_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "state_word_count",
                                  state_float_count
                                )},
        {"source", std::string {shadow_state_source}},
        {"capture", std::string {shadow_state_capture}},
        {"rendered_output_selected", true},
        {"wire_contract", "authenticated live Host-SBS renderer input; not a client wire contract"},
        {"units", {
                    {"coordinate", "dimensionless canonical coordinate derived from raw depth"},
                    {"gain", completed.depth_input_region.video_region ?
                      "one-eye ROI-local source-U per curve unit" :
                      "one-eye full-source-U per curve unit"},
                    {"parallax", completed.depth_input_region.video_region ?
                      "signed one-eye ROI-local source-U; full-source renderer authority additionally requires depth_input_region embedding" :
                      "signed one-eye full-source-U"},
                  }},
        {"constants", {
                        {"raw_coordinate_scale", completed.parallax_v2_raw_coordinate_scale},
                        {"collapse_abs_epsilon", collapse_abs_epsilon},
                        {"far_tau", far_tau},
                        {"near_log_tau", near_log_tau},
                        {"gain_per_pop", gain_per_pop},
                        {"reference_pop_strength", reference_pop_strength},
                        {"reference_gain_at_reference_pop", parallax_gain},
                        {"requested_gain", completed.parallax_v2_requested_gain},
                        {"requested_pop_strength", completed.parallax_v2_requested_pop_strength},
                        {"direct_container_limit", direct_container_limit},
                        {"max_horizontal_slope", max_horizontal_slope},
                        {"max_vertical_shear", max_vertical_shear},
                        {"vertical_majorant_share", vertical_majorant_share},
                        {"convergence_curve_default", convergence_curve_default},
                      }},
        {"fields", std::move(fields)},
        {"named_values", std::move(named_values)},
        {"decoded", decoded},
        {"adaptation_semantics", {
                                    {"coordinate", "immediate-first-usable-center-latched-until-cut-fixed-authenticated-scale-retained-across-unusable"},
                                    {"convergence_curve", "arithmetic-mean-center-is-zero-plane"},
                                    {"requested_gain", "immutable-cfg-pop-strength"},
                                    {"container_scale", "abi-retained-identity-pointwise-soft-container-is-map-local"},
                                    {"near_curve", "fixed-contract-logarithmic-tau-independent-of-content-occupancy"},
                                    {"spatial_conditioner", "fixed-75pct-vertical-majorant-share-then-horizontal-majorant"},
                                  }},
      };

      nlohmann::json frame_named = nlohmann::json::object();
      for (std::size_t index = 0; index < frame_stats.size(); ++index) {
        frame_named[frame_stat_names[index]] = frame_stats[index];
      }
      const nlohmann::json frame_json {
        {"schema", shadow_frame_stats_dump_schema},
        {"coordinate_contract", parallax_v2_coordinate_binding(
                                  shader_identity,
                                  "frame_stats_word_count",
                                  frame_stats_float_count
                                )},
        {"source", std::string {frame_stats_source}},
        {"named_values", std::move(frame_named)},
      };
      summary = decoded;
      summary["raw_coordinate_scale"] = completed.parallax_v2_raw_coordinate_scale;
      summary["rendered_output_selected"] = true;
      return write_json(dir / "shadow_state.json", state_json) &&
             write_json(dir / "shadow_frame_stats.json", frame_json);
    }

    depth_dumpability decode_normalization_state(
      const std::vector<std::uint8_t> &bytes,
      normalization_state &state
    ) {
      if (bytes.size() != 4u * sizeof(float)) {
        return depth_dumpability::unreadable;
      }
      std::array<float, 4> values {};
      std::memcpy(values.data(), bytes.data(), bytes.size());
      for (const float value : values) {
        if (!std::isfinite(value)) {
          return depth_dumpability::unreadable;
        }
      }
      state = {values[0], values[1], values[2], values[3]};
      return state.frame_state >= 0.5f ?
               depth_dumpability::valid :
               depth_dumpability::invalid;
    }

    bool shadow_state_is_dumpable(
      const std::vector<float> &state,
      const float raw_coordinate_scale
    ) {
      using namespace models::depth_coordinate_v2;
      if (state.size() != state_float_count) {
        return false;
      }
      state_words_t words {};
      std::memcpy(words.data(), state.data(), sizeof(words));
      return words[frame_valid] == std::bit_cast<std::uint32_t>(1.0f) &&
             parallax_state_words_are_authenticated(words, raw_coordinate_scale);
    }

    const char *encoding_name(const sbs_adaptive_state::gpu_encoding_e encoding) {
      switch (encoding) {
        case sbs_adaptive_state::gpu_encoding_e::float_value:
          return "float_value";
        case sbs_adaptive_state::gpu_encoding_e::uint_bits:
          return "uint_bits";
        case sbs_adaptive_state::gpu_encoding_e::uint_valued_float:
          return "uint_valued_float";
      }
      return "unknown";
    }

    bool dump_adaptive_state(
      const std::vector<std::uint8_t> &bytes,
      const normalization_state &normalization,
      const frame &completed,
      const std::filesystem::path &dir,
      nlohmann::json &adaptive_summary
    ) {
      if (bytes.size() !=
          sbs_adaptive_state::word_count * sizeof(std::uint32_t)) {
        return false;
      }
      sbs_adaptive_state::words_t words {};
      std::memcpy(words.data(), bytes.data(), bytes.size());
      nlohmann::json fields = nlohmann::json::array();
      nlohmann::json values = nlohmann::json::array();
      nlohmann::json named_values = nlohmann::json::object();
      std::array<float, sbs_adaptive_state::word_count> scalars {};
      for (const auto &descriptor : sbs_adaptive_state::fields) {
        const std::size_t index = sbs_adaptive_state::index(descriptor.word);
        const std::uint32_t raw_word = words[index];
        nlohmann::json value;
        if (
          descriptor.gpu_encoding ==
          sbs_adaptive_state::gpu_encoding_e::uint_bits
        ) {
          value = raw_word;
        } else {
          const float scalar = std::bit_cast<float>(raw_word);
          if (!std::isfinite(scalar)) {
            return false;
          }
          scalars[index] = scalar;
          if (
            descriptor.gpu_encoding ==
            sbs_adaptive_state::gpu_encoding_e::uint_valued_float
          ) {
            if (
              scalar < 0.0f ||
              scalar > static_cast<float>(std::numeric_limits<std::uint32_t>::max()) ||
              std::trunc(scalar) != scalar
            ) {
              return false;
            }
            value = static_cast<std::uint32_t>(scalar);
          } else {
            value = scalar;
          }
        }
        values.push_back(value);
        named_values[std::string(descriptor.name)] = value;
        fields.push_back({
          {"index", index},
          {"name", std::string(descriptor.name)},
          {"json_type", std::string(descriptor.json_type)},
          {"gpu_encoding", encoding_name(descriptor.gpu_encoding)},
          {"raw_word", raw_word},
          {"value", std::move(value)},
        });
      }

      using sbs_adaptive_state::word_e;
      const auto scalar = [&](const word_e word) {
        return scalars[sbs_adaptive_state::index(word)];
      };
      const float cut_flags_value = scalar(word_e::cut_flags);
      const float analysis_flags_value = scalar(word_e::analysis_flags);
      if (
        cut_flags_value < 0.0f ||
        cut_flags_value >
          static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) ||
        std::trunc(cut_flags_value) != cut_flags_value ||
        analysis_flags_value < 0.0f ||
        analysis_flags_value >
          static_cast<float>(sbs_adaptive_state::known_analysis_flag_mask) ||
        std::trunc(analysis_flags_value) != analysis_flags_value
      ) {
        return false;
      }
      const auto cut_flags = static_cast<std::uint32_t>(cut_flags_value);
      const auto analysis_flags =
        static_cast<std::uint32_t>(analysis_flags_value);
      nlohmann::json decoded_cut_flags = nlohmann::json::object();
      for (const auto &bit : sbs_adaptive_state::cut_flag_bits) {
        decoded_cut_flags[std::string(bit.name)] = (cut_flags & bit.mask) != 0u;
      }
      nlohmann::json decoded_analysis_flags = nlohmann::json::object();
      for (const auto &bit : sbs_adaptive_state::analysis_flag_bits) {
        decoded_analysis_flags[std::string(bit.name)] =
          (analysis_flags & bit.mask) != 0u;
      }
      adaptive_summary = {
        {"schema", sbs_adaptive_state::schema_version},
        {"source", std::string(sbs_adaptive_state::source)},
        {"capture", std::string(sbs_adaptive_state::capture)},
        {"role", "comparison-only scene-cut bridge evidence; no live V2 geometry authority"},
        {"matched_frame_id", completed.matched_frame_id},
        {"depth_model", completed.depth_model},
        {"fields", std::move(fields)},
        {"values", std::move(values)},
        {"named_values", std::move(named_values)},
        {"decoded", {
                      {"cut_flags", {
                                      {"value", cut_flags},
                                      {"known_mask", sbs_adaptive_state::known_cut_flag_mask},
                                      {"bits", std::move(decoded_cut_flags)},
                                    }},
                      {"analysis_flags", {
                                           {"value", analysis_flags},
                                           {"known_mask", sbs_adaptive_state::known_analysis_flag_mask},
                                           {"bits", std::move(decoded_analysis_flags)},
                                         }},
                      {"hard_cut_pulse", scalar(word_e::hard_cut_pulse) > 0.5f},
                      {"hard_cut_count", words[sbs_adaptive_state::index(word_e::hard_cut_count)]},
                      {"reserved_cut_bridge_17", words[sbs_adaptive_state::index(word_e::reserved_cut_bridge_17)]},
                      {"empty_raw_count", words[sbs_adaptive_state::index(word_e::empty_raw_count)]},
                      {"collapsed_raw_count", words[sbs_adaptive_state::index(word_e::collapsed_raw_count)]},
                      {"geometry_authority", false},
                    }},
        {"normalization", {
                            {"role", "comparison-only scene-cut bridge evidence"},
                            {"effective_lower", normalization.lower},
                            {"effective_upper", normalization.upper},
                            {"initialized", normalization.initialized > 0.5f},
                            {"initialized_value", normalization.initialized},
                            {"frame_state", normalization_frame_state_name(normalization.frame_state)},
                            {"frame_state_value", normalization.frame_state},
                          }},
      };
      return write_json(dir / "adaptive_state.json", adaptive_summary);
    }

    bool dump_warp_map(
      const texture_snapshot &mapping,
      const std::uint32_t source_width,
      const std::uint32_t source_height,
      const bool video_region,
      const std::filesystem::path &dir,
      warp_map_dump_stats &stats
    ) {
      std::vector<float> map;
      if (
        !texture_float_values(mapping, map) ||
        mapping.desc.Width < 2u || (mapping.desc.Width & 1u) != 0u ||
        source_width == 0u || source_height == 0u
      ) {
        return false;
      }
      stats.width = mapping.desc.Width;
      stats.height = mapping.desc.Height;
      stats.eye_width = mapping.desc.Width / 2u;
      stats.eye_height = mapping.desc.Height;
      const float source_aspect =
        static_cast<float>(source_width) / static_cast<float>(source_height);
      const float eye_aspect =
        static_cast<float>(stats.eye_width) / static_cast<float>(stats.eye_height);
      stats.content_scale_x =
        eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
      stats.content_scale_y =
        eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;
      static_cast<scalar_stats &>(stats) = calculate_scalar_stats(map);

      std::vector<float> displacement(map.size(), 0.0f);
      std::vector<float> finite_absolute_displacement;
      finite_absolute_displacement.reserve(map.size());
      std::vector<std::uint8_t> content_valid(map.size(), 0u);
      const float content_lo_x = 0.5f * (1.0f - stats.content_scale_x);
      const float content_hi_x = content_lo_x + stats.content_scale_x;
      const float content_lo_y = 0.5f * (1.0f - stats.content_scale_y);
      const float content_hi_y = content_lo_y + stats.content_scale_y;
      for (std::uint32_t y = 0; y < stats.height; ++y) {
        const float output_v =
          (static_cast<float>(y) + 0.5f) / static_cast<float>(stats.height);
        for (std::uint32_t x = 0; x < stats.width; ++x) {
          const std::size_t index =
            static_cast<std::size_t>(y) * stats.width + x;
          const std::uint32_t eye_x = x % stats.eye_width;
          const float output_u =
            (static_cast<float>(eye_x) + 0.5f) /
            static_cast<float>(stats.eye_width);
          if (
            output_u < content_lo_x || output_u > content_hi_x ||
            output_v < content_lo_y || output_v > content_hi_y ||
            !std::isfinite(map[index])
          ) {
            continue;
          }
          const float unwarped_source_u =
            (output_u - content_lo_x) / stats.content_scale_x;
          displacement[index] =
            (map[index] - unwarped_source_u) *
            stats.content_scale_x *
            static_cast<float>(stats.eye_width);
          if (std::isfinite(displacement[index])) {
            content_valid[index] = 1u;
            finite_absolute_displacement.push_back(std::fabs(displacement[index]));
          }
        }
      }
      if (finite_absolute_displacement.empty()) {
        return false;
      }
      std::sort(
        finite_absolute_displacement.begin(),
        finite_absolute_displacement.end()
      );
      const std::size_t p98_index = static_cast<std::size_t>(
        std::lround(
          0.98 * static_cast<double>(finite_absolute_displacement.size() - 1u)
        )
      );
      stats.displacement_preview_abs_px =
        finite_absolute_displacement[std::min(
          p98_index,
          finite_absolute_displacement.size() - 1u
        )];
      if (!(stats.displacement_preview_abs_px > 1.0e-6f)) {
        stats.displacement_preview_abs_px =
          finite_absolute_displacement.back();
      }
      if (!(stats.displacement_preview_abs_px > 1.0e-6f)) {
        stats.displacement_preview_abs_px = 1.0f;
      }

      std::vector<std::uint8_t> heat(map.size() * 3u, 0u);
      for (std::size_t index = 0; index < map.size(); ++index) {
        if (!content_valid[index]) {
          if (!std::isfinite(map[index])) {
            heat[index * 3u + 0u] = 255;
            heat[index * 3u + 1u] = 0;
            heat[index * 3u + 2u] = 255;
          }
          continue;
        }
        const float normalized = std::clamp(
          0.5f +
            0.5f * displacement[index] / stats.displacement_preview_abs_px,
          0.0f,
          1.0f
        );
        colormap_jet(
          normalized,
          heat[index * 3u + 0u],
          heat[index * 3u + 1u],
          heat[index * 3u + 2u]
        );
      }
      const nlohmann::json shape {
        {"schema", 2},
        {"width", stats.width},
        {"height", stats.height},
        {"eye_width", stats.eye_width},
        {"eye_height", stats.eye_height},
        {"source_width", source_width},
        {"source_height", source_height},
        {"content_scale_x", stats.content_scale_x},
        {"content_scale_y", stats.content_scale_y},
        {"dtype", "float32-le"},
        {"layout", "row-major"},
        {"channels", {"raw_reproject_source_u_normalized"}},
        {"validity", {
          {"content", "derive from content_scale_x/content_scale_y and packed output coordinate"},
          {"inverse", video_region ?
            "11-step contractive fixed-point solution of crop-local q embedded by depth_input_region.json scale and outside-only zero-plane collar" :
            "11-step contractive fixed-point solution of the signed final-parallax field"},
          {"mask", "warp_mask.png red marks finite-source boundary extrapolation; V2 has no internal owner or synthetic-fill path"},
        }},
        {"live_sample_source_u_normalized", "clamp(raw_reproject_source_u_normalized, 0, 1)"},
        {"derived_inverse_displacement_output_eye_px", "(raw_reproject_source_u_normalized - aspect_fitted_unwarped_source_u) * content_scale_x * eye_width"},
        {"derived_signed_binocular_disparity_px", "invert both eye maps at common source-U samples; x_right - x_left"},
        {"displacement_preview", {
                                   {"file", "warp_displacement_heat.png"},
                                   {"range_px", {
                                                  -stats.displacement_preview_abs_px,
                                                  stats.displacement_preview_abs_px,
                                                }},
                                   {"normalization", "symmetric finite-content p98 absolute displacement"},
                                   {"negative", "blue"},
                                   {"zero", "green"},
                                   {"positive", "red"},
                                   {"bars", "black"},
                                   {"nonfinite", "magenta"},
                                 }},
      };
      return write_bytes(
               dir / "warp_map.f32",
               map.data(),
               map.size() * sizeof(float)
             ) &&
             write_png(
               dir / "warp_displacement_heat.png",
               stats.width,
               stats.height,
               heat
             ) &&
             write_json(dir / "warp_map_shape.json", shape);
    }

    nlohmann::json config_json(
      const config::video_t::sbs_t &cfg,
      const frame &completed,
      const std::string &model_name,
      const std::string &effective_model_url
    ) {
      return {
        {"schema", 3},
        {"shared_configured", {
          {"pop_strength", cfg.pop_strength},
          {"max_packed_encode_width", cfg.max_encode_width},
        }},
        {"live_effective", {
          {"renderer", "depth-coordinate-v2"},
          {"pop_strength", completed.parallax_v2_requested_pop_strength},
          {"adaptive_pop", false},
          {"zero_plane_authority", "scene-latched selected raw center"},
          {"depth_model", model_name},
          {"depth_model_url", effective_model_url},
          {"model_input_width", completed.model_width},
          {"model_input_height", completed.model_height},
          {"cuda_graph_active", completed.cuda_graph_active},
          {"cut_analysis", {
            {"mode", "cut-only"},
            {"depth_ema", config::host_sbs_v2_live_calibration::depth_ema},
            {"ema_edge_change", config::host_sbs_v2_live_calibration::edge_change},
            {"ema_edge_gradient", config::host_sbs_v2_live_calibration::edge_gradient},
            {"ema_edge_strength", config::host_sbs_v2_live_calibration::edge_strength},
            {"minmax_ema", config::host_sbs_v2_live_calibration::minmax_ema},
          }},
        }},
      };
    }

    nlohmann::json artifact_description(
      const bool available,
      const bool required,
      const std::string_view stage,
      const std::string_view description
    ) {
      return {
        {"available", available},
        {"required", required},
        {"stage", std::string(stage)},
        {"description", std::string(description)},
      };
    }

    nlohmann::json hashed_artifact_description(
      const bool available,
      const bool required,
      const std::string_view stage,
      const std::string_view description,
      const std::string &sha256
    ) {
      auto descriptor = artifact_description(available, required, stage, description);
      descriptor["sha256"] = sha256;
      return descriptor;
    }

    const char *gpu_trace_submission_class_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::submission_class_e;
      switch (static_cast<submission_class_e>(value)) {
        case submission_class_e::force_infer:
          return "force-infer";
        case submission_class_e::gpu_undecided:
          return "gpu-undecided";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_depth_disposition_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::depth_disposition_e;
      switch (static_cast<depth_disposition_e>(value)) {
        case depth_disposition_e::reuse:
          return "reuse";
        case depth_disposition_e::infer:
          return "infer";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_subtitle_disposition_name(
      const std::uint32_t value
    ) noexcept {
      using models::host_sbs_gpu_trace::subtitle_disposition_e;
      switch (static_cast<subtitle_disposition_e>(value)) {
        case subtitle_disposition_e::suppressed:
          return "suppressed";
        case subtitle_disposition_e::optional_ocr:
          return "optional-ocr";
        case subtitle_disposition_e::abstention:
          return "abstention";
        case subtitle_disposition_e::held_with_depth:
          return "held-with-depth";
        default:
          return "invalid";
      }
    }

    const char *gpu_trace_host_outcome_name(const std::uint32_t value) noexcept {
      using models::host_sbs_gpu_trace::host_subtitle_outcome_e;
      switch (static_cast<host_subtitle_outcome_e>(value)) {
        case host_subtitle_outcome_e::suppressed:
          return "suppressed";
        case host_subtitle_outcome_e::ordinary_record:
          return "ordinary-record";
      }
      return "invalid";
    }

    const char *gpu_trace_expected_work_name(const std::uint32_t value) noexcept {
      using cuda_conditional_graph::work_flag_e;
      using cuda_conditional_graph::work_flags_value;
      if (value == work_flags_value(work_flag_e::none)) {
        return "none";
      }
      if (value == work_flags_value(work_flag_e::optional_ocr)) {
        return "optional-ocr";
      }
      if (value == work_flags_value(work_flag_e::subtitle_observation)) {
        return "subtitle-observation";
      }
      if (value == work_flags_value(work_flag_e::optional_ocr_due)) {
        return "optional-ocr-due";
      }
      if (value == work_flags_value(work_flag_e::subtitle_observation_due)) {
        return "subtitle-observation-due";
      }
      return "invalid";
    }

    const char *gpu_trace_locator_event_name(const std::uint32_t value) noexcept {
      switch (value) {
        case 0u:
          return "none";
        case 1u:
          return "birth";
        case 2u:
          return "death";
        case 3u:
          return "handoff";
        default:
          return "invalid";
      }
    }

    std::uint32_t gpu_trace_word(
      const std::vector<std::uint8_t> &ring,
      const std::size_t index
    ) noexcept {
      std::uint32_t value = 0u;
      if (index < ring.size() / sizeof(value)) {
        std::memcpy(&value, ring.data() + index * sizeof(value), sizeof(value));
      }
      return value;
    }

    nlohmann::json gpu_trace_word_array(
      const std::vector<std::uint8_t> &ring,
      const std::size_t begin,
      const std::size_t count
    ) {
      auto result = nlohmann::json::array();
      for (std::size_t index = 0u; index < count; ++index) {
        result.push_back(gpu_trace_word(ring, begin + index));
      }
      return result;
    }

    nlohmann::json gpu_trace_contract_document(
      const models::host_sbs_gpu_trace_provenance_t &provenance
    ) {
      using namespace models::host_sbs_gpu_trace;
      using models::host_sbs_shader_cache::host_sbs_gpu_trace;
      using cuda_conditional_graph::work_flag_e;
      using cuda_conditional_graph::work_flags_value;
      const auto offset = [](const auto value) {
        return static_cast<std::uint32_t>(value);
      };
      return {
        {"schema", dump_contract_schema},
        {"role", "diagnostic-only accepted-root completion history; never rendering authority"},
        {"byte_order", "little-endian"},
        {"decoded_trace_schema", decoded_trace_schema},
        {"ring", {
          {"schema", ring_schema},
          {"tag", ring_tag},
          {"capacity", capacity},
          {"header_words", header_word_count},
          {"record_words", record_word_count},
          {"record_bytes", record_word_count * sizeof(std::uint32_t)},
          {"ring_words", ring_word_count},
          {"ring_bytes", ring_byte_count},
          {"record_array_word_offset", header_word_count},
          {"record_array_byte_offset", header_word_count * sizeof(std::uint32_t)},
          {"record_slot_word_stride", record_word_count},
          {"record_slot_byte_stride", record_word_count * sizeof(std::uint32_t)},
          {"commit_protocol", "header tag invalidated before slot overwrite; payload then record tag; cursor then header tag last"},
        }},
        {"header_word_offsets", {
          {"schema", offset(header_word_e::schema)},
          {"tag", offset(header_word_e::tag)},
          {"capacity", offset(header_word_e::capacity)},
          {"record_words", offset(header_word_e::record_words)},
          {"next_sequence_low", offset(header_word_e::next_sequence_low)},
          {"next_sequence_high", offset(header_word_e::next_sequence_high)},
          {"next_slot", offset(header_word_e::next_slot)},
          {"committed_count", offset(header_word_e::committed_count)},
          {"reserved_begin", offset(header_word_e::reserved_begin)},
          {"end", header_word_count},
        }},
        {"record_word_offsets", {
          {"schema", offset(record_word_e::schema)},
          {"commit_tag", offset(record_word_e::commit_tag)},
          {"sequence_low", offset(record_word_e::sequence_low)},
          {"sequence_high", offset(record_word_e::sequence_high)},
          {"frame_low", offset(record_word_e::frame_low)},
          {"frame_high", offset(record_word_e::frame_high)},
          {"analysis_generation_low", offset(record_word_e::analysis_generation_low)},
          {"analysis_generation_high", offset(record_word_e::analysis_generation_high)},
          {"domain_tag_low", offset(record_word_e::domain_tag_low)},
          {"domain_tag_high", offset(record_word_e::domain_tag_high)},
          {"transaction_token_low", offset(record_word_e::transaction_token_low)},
          {"transaction_token_high", offset(record_word_e::transaction_token_high)},
          {"submission_class", offset(record_word_e::submission_class)},
          {"depth_disposition", offset(record_word_e::depth_disposition)},
          {"expected_work", offset(record_word_e::expected_work)},
          {"subtitle_disposition", offset(record_word_e::subtitle_disposition)},
          {"flags", offset(record_word_e::flags)},
          {"host_subtitle_outcome", offset(record_word_e::host_subtitle_outcome)},
          {"source_width", offset(record_word_e::source_width)},
          {"source_height", offset(record_word_e::source_height)},
          {"field_width", offset(record_word_e::field_width)},
          {"field_height", offset(record_word_e::field_height)},
          {"transaction_words", offset(record_word_e::transaction_words)},
          {"reserved0", offset(record_word_e::reserved0)},
          {"transaction_begin", offset(record_word_e::transaction_begin)},
          {"subtitle_locator_begin", offset(record_word_e::subtitle_locator_begin)},
          {"subtitle_condition_begin", offset(record_word_e::subtitle_condition_begin)},
          {"observation_timestamp_low", offset(record_word_e::observation_timestamp_low)},
          {"observation_timestamp_high", offset(record_word_e::observation_timestamp_high)},
          {"reserved_begin", offset(record_word_e::reserved_begin)},
          {"end", record_word_count},
        }},
        {"record_sections", {
          {"transaction", {
            {"word_offset", offset(record_word_e::transaction_begin)},
            {"word_count", transaction_word_count},
            {"validity", "immutable postprocessed transaction snapshot; branch remains invalid unless RQST/CBRG/cookies/token/class all authenticate"},
          }},
          {"subtitle_locator", {
            {"word_offset", offset(record_word_e::subtitle_locator_begin)},
            {"word_count", subtitle_locator_word_count},
            {"validity", "post-finalization SLR13 for an authenticated subtitle observation; held_with_depth preserves the byte-identical immediately prior tuple and frame identity; subtitle_suppressed bytes are frozen/unused"},
          }},
          {"subtitle_condition", {
            {"word_offset", offset(record_word_e::subtitle_condition_begin)},
            {"word_count", subtitle_condition_word_count},
            {"validity", "post-finalization condition params for an authenticated subtitle observation, or the byte-identical immediately prior params for held_with_depth: exact SLR tuple when current_count is nonzero, canonical zero6 when current_count is zero; suppressed output copies exact Base and these words are unused"},
          }},
          {"observation_timestamp", {
            {"word_offset", offset(record_word_e::observation_timestamp_low)},
            {"word_count", 2u},
            {"validity", "nonzero monotonic source-observation microseconds; zero is invalid"},
          }},
          {"reserved", {
            {"word_offset", offset(record_word_e::reserved_begin)},
            {"word_count", record_word_count - offset(record_word_e::reserved_begin)},
            {"validity", "must be zero"},
          }},
        }},
        {"transaction_word_offsets", {
          {"receipt", {{"word_offset", 0u}, {"word_count", 8u}}},
          {"request", {{"word_offset", 8u}, {"word_count", 8u}}},
          {"indirect_dispatch_groups", {{"word_offset", 16u}, {"word_count", 48u}, {"group_word_stride", 4u}}},
        }},
        {"trace_constant_word_offsets", {
          {"frame_low", 0u}, {"frame_high", 1u},
          {"analysis_generation_low", 2u}, {"analysis_generation_high", 3u},
          {"domain_tag_low", 4u}, {"domain_tag_high", 5u},
          {"transaction_token_low", 6u}, {"transaction_token_high", 7u},
          {"expected_work", 8u}, {"submission_class", 9u},
          {"flags", 10u}, {"host_subtitle_outcome", 11u},
          {"source_width", 12u}, {"source_height", 13u},
          {"field_width", 14u}, {"field_height", 15u},
          {"observation_timestamp_low", 16u}, {"observation_timestamp_high", 17u},
          {"padding0", 18u}, {"padding1", 19u},
        }},
        {"tags", {{"header", ring_tag}, {"record", record_tag}}},
        {"receipt_abi", {
          {"decision_cookie", cuda_conditional_graph::decision_cookie},
          {"token_low_cookie", cuda_conditional_graph::token_low_cookie},
          {"token_high_cookie", cuda_conditional_graph::token_high_cookie},
          {"work_flags_cookie", cuda_conditional_graph::work_flags_cookie},
          {"proposal_magic", cuda_conditional_graph::proposal_magic},
          {"request_magic", cuda_conditional_graph::request_magic},
          {"receipt_magic", cuda_conditional_graph::receipt_magic},
          {"optional_ocr_receipt_magic", cuda_conditional_graph::optional_ocr_receipt_magic},
          {"branch", {{"reuse", 0u}, {"infer", 1u}}},
          {"work", {
            {"none", work_flags_value(work_flag_e::none)},
            {"optional_ocr", work_flags_value(work_flag_e::optional_ocr)},
            {"subtitle_observation", work_flags_value(work_flag_e::subtitle_observation)},
            {"optional_ocr_due", work_flags_value(work_flag_e::optional_ocr_due)},
            {"subtitle_observation_due", work_flags_value(work_flag_e::subtitle_observation_due)},
          }},
        }},
        {"enums", {
          {"submission_class", {{"invalid", 0u}, {"force_infer", 1u}, {"gpu_undecided", 2u}}},
          {"depth_disposition", {{"invalid", 0u}, {"reuse", 1u}, {"infer", 2u}}},
          {"subtitle_disposition", {
            {"suppressed", 0u}, {"optional_ocr", 1u}, {"abstention", 2u},
            {"held_with_depth", 5u}, {"invalid", 6u},
          }},
          {"host_subtitle_outcome", {
            {"suppressed", 0u}, {"ordinary_record", 1u},
          }},
          {"flags", {
            {"input_domain_reset", input_domain_reset},
            {"dump_forced_at_enqueue", dump_forced},
            {"ocr_record_submitted", ocr_record_submitted},
            {"subtitle_suppressed", subtitle_suppressed},
            {"condition_executed", condition_executed},
            {"subtitle_branch_gated", subtitle_branch_gated},
            {"known_mask", known_record_flags},
          }},
        }},
        {"subtitle_locator", {
          {"schema", subtitle_locator_state_schema},
          {"tag", subtitle_locator_state_tag},
          {"word_count", subtitle_locator_word_count},
        }},
        {"subtitle_condition", {
          {"schema", subtitle_condition_param_schema},
          {"tag", subtitle_condition_param_tag},
          {"word_count", subtitle_condition_word_count},
        }},
        {"shader", {
          {"source_file", std::string {host_sbs_gpu_trace.filename}},
          {"entrypoint", std::string {host_sbs_gpu_trace.entrypoint}},
          {"profile", std::string {host_sbs_gpu_trace.target}},
          {"source_closure_schema", provenance.source_closure_schema},
          {"source_compile_flags", provenance.source_compile_flags},
          {"source_macro_count", provenance.source_macro_count},
          {"source_closure_sha256", provenance.source_closure_sha256},
        }},
      };
    }

    nlohmann::json gpu_trace_decoded_document(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      using namespace models::host_sbs_gpu_trace;
      const auto next_sequence = join_u64(
        gpu_trace_word(ring, word_index(header_word_e::next_sequence_low)),
        gpu_trace_word(ring, word_index(header_word_e::next_sequence_high))
      );
      const auto next_slot = gpu_trace_word(
        ring, word_index(header_word_e::next_slot)
      );
      const auto count = gpu_trace_word(
        ring, word_index(header_word_e::committed_count)
      );
      const auto oldest_sequence = next_sequence - count;
      const auto oldest_slot = (next_slot + capacity - count) % capacity;
      auto records = nlohmann::json::array();
      std::uint64_t matched_sequence = 0u;
      for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        const auto slot = (oldest_slot + ordinal) % capacity;
        const auto base = record_base(slot);
        const auto word = [&ring, base](const record_word_e field) {
          return gpu_trace_word(ring, base + word_index(field));
        };
        const auto sequence = join_u64(
          word(record_word_e::sequence_low), word(record_word_e::sequence_high)
        );
        const auto frame_id = join_u64(
          word(record_word_e::frame_low), word(record_word_e::frame_high)
        );
        const auto analysis_generation = join_u64(
          word(record_word_e::analysis_generation_low),
          word(record_word_e::analysis_generation_high)
        );
        const auto domain_tag = join_u64(
          word(record_word_e::domain_tag_low), word(record_word_e::domain_tag_high)
        );
        const auto token = join_u64(
          word(record_word_e::transaction_token_low),
          word(record_word_e::transaction_token_high)
        );
        const auto flags = word(record_word_e::flags);
        const auto locator_base =
          base + word_index(record_word_e::subtitle_locator_begin);
        const auto condition_base =
          base + word_index(record_word_e::subtitle_condition_begin);
        const auto locator_word = [&ring, locator_base](const std::size_t index) {
          return gpu_trace_word(ring, locator_base + index);
        };
        const auto condition_word = [&ring, condition_base](const std::size_t index) {
          return gpu_trace_word(ring, condition_base + index);
        };
        const auto locator_flags = locator_word(2u);
        const auto owner_count = locator_word(4u);
        const auto lifetime_or_hold = locator_word(25u);
        const auto target_bits = locator_word(18u);
        const auto target = std::bit_cast<float>(target_bits);
        const auto event = locator_word(21u);
        const bool suppressed = (flags & subtitle_suppressed) != 0u;
        const bool condition_valid = !suppressed &&
          condition_word(0u) == subtitle_condition_param_schema &&
          condition_word(1u) == subtitle_condition_param_tag;
        const auto condition_target_bits = condition_word(5u);
        const auto condition_target = std::bit_cast<float>(condition_target_bits);
        const bool condition_executed_host_proven =
          (flags & condition_executed) != 0u;
        const bool branch_gated = (flags & subtitle_branch_gated) != 0u;
        const auto subtitle_disposition = static_cast<subtitle_disposition_e>(
          word(record_word_e::subtitle_disposition)
        );
        const bool subtitle_held_with_depth =
          subtitle_disposition == subtitle_disposition_e::held_with_depth;
        const bool subtitle_observation_executed =
          subtitle_disposition == subtitle_disposition_e::optional_ocr ||
          subtitle_disposition == subtitle_disposition_e::abstention;
        const bool condition_was_executed = condition_executed_host_proven ||
          (branch_gated && subtitle_observation_executed);
        const bool condition_was_published =
          condition_was_executed || subtitle_held_with_depth;
        const bool condition_active = condition_was_published && condition_valid &&
          condition_word(2u) != 0u &&
          (condition_word(4u) == 1u || condition_word(4u) == 2u) &&
          std::isfinite(condition_target) &&
          std::fabs(condition_target) <= direct_container_limit;
        const bool matched = frame_id == completed.matched_frame_id;
        if (matched) {
          matched_sequence = sequence;
        }
        records.push_back({
          {"sequence", sequence},
          {"ring_slot", slot},
          {"frame_id", frame_id},
          {"analysis_generation", analysis_generation},
          {"domain_tag", domain_tag},
          {"transaction_token", token},
          {"observation_timestamp_us", join_u64(
            word(record_word_e::observation_timestamp_low),
            word(record_word_e::observation_timestamp_high)
          )},
          {"submission_class", {
            {"value", word(record_word_e::submission_class)},
            {"name", gpu_trace_submission_class_name(word(record_word_e::submission_class))},
          }},
          {"depth_disposition", {
            {"value", word(record_word_e::depth_disposition)},
            {"name", gpu_trace_depth_disposition_name(word(record_word_e::depth_disposition))},
          }},
          {"expected_work", {
            {"value", word(record_word_e::expected_work)},
            {"name", gpu_trace_expected_work_name(word(record_word_e::expected_work))},
          }},
          {"subtitle_disposition", {
            {"value", word(record_word_e::subtitle_disposition)},
            {"name", gpu_trace_subtitle_disposition_name(word(record_word_e::subtitle_disposition))},
          }},
          {"host_subtitle_outcome", {
            {"value", word(record_word_e::host_subtitle_outcome)},
            {"name", gpu_trace_host_outcome_name(word(record_word_e::host_subtitle_outcome))},
          }},
          {"flags", {
            {"value", flags},
            {"input_domain_reset", (flags & input_domain_reset) != 0u},
            {"dump_forced_at_enqueue", (flags & dump_forced) != 0u},
            {"ocr_record_submitted", (flags & ocr_record_submitted) != 0u},
            {"subtitle_suppressed", (flags & subtitle_suppressed) != 0u},
            {"condition_executed", condition_was_executed},
            {"condition_executed_host_proven", condition_executed_host_proven},
            {"subtitle_branch_gated", branch_gated},
          }},
          {"analysis_source", {
            {"width", word(record_word_e::source_width)},
            {"height", word(record_word_e::source_height)},
          }},
          {"field", {
            {"width", word(record_word_e::field_width)},
            {"height", word(record_word_e::field_height)},
          }},
          {"matched_dump_frame", matched},
          {"subtitle_locator", {
            {"schema", locator_word(0u)},
            {"tag", locator_word(1u)},
            {"flags", {
              {"value", locator_flags},
              {"owner", (locator_flags & 1u) != 0u},
              {"pending", (locator_flags & 2u) != 0u},
              {"target_valid", (locator_flags & 4u) != 0u},
              {"target_reset", (locator_flags & 8u) != 0u},
              {"provisional_current",
               (locator_flags & subtitle_locator_flag_provisional_current) != 0u},
            }},
            {"owner_generation", locator_word(3u)},
            {"owner_count", owner_count},
            {"pending_count", locator_word(12u)},
            {"current_count", locator_word(20u)},
            {"target", target},
            {"target_bits", target_bits},
            {"event", {{"value", event}, {"name", gpu_trace_locator_event_name(event)}}},
            {"fade_step", locator_word(24u)},
            {"provisional_target",
             (locator_flags & subtitle_locator_flag_provisional_current) != 0u ?
               nlohmann::json(std::bit_cast<float>(locator_word(
                 subtitle_locator_provisional_target_word
               ))) : nlohmann::json(nullptr)},
            {"provisional_target_bits", locator_word(
              subtitle_locator_provisional_target_word
            )},
            {"provisional_fade_step",
             (locator_flags & subtitle_locator_flag_provisional_current) != 0u ?
               locator_word(subtitle_locator_provisional_fade_word) : 0u},
            {"lifetime_or_hold_count", lifetime_or_hold},
            {"unreliable_hold_count", owner_count != 0u ?
              nlohmann::json(lifetime_or_hold) : nlohmann::json(nullptr)},
            {"death_grace_count", owner_count == 0u ?
              nlohmann::json(lifetime_or_hold) : nlohmann::json(nullptr)},
            {"cut_epoch", locator_word(26u)},
            {"analysis_generation", join_u64(locator_word(10u), locator_word(11u))},
            {"frame_id", join_u64(locator_word(22u), locator_word(23u))},
          }},
          {"subtitle_condition", suppressed ?
            nlohmann::json {
              {"active", false},
              {"unused", true},
              {"reason", "subtitle-suppressed transaction freezes SLR and skips conditioning"},
            } :
            nlohmann::json {
              {"active", condition_active},
              {"valid", condition_valid},
              {"executed", condition_was_executed},
              {"held_with_depth", subtitle_held_with_depth},
              {"unused", false},
              {"schema", condition_word(0u)},
              {"tag", condition_word(1u)},
              {"current_count", condition_word(2u)},
              {"current_kinds", condition_word(3u)},
              {"fade_step", condition_word(4u)},
              {"target", condition_target},
              {"target_bits", condition_target_bits},
            }},
          {"transaction_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::transaction_begin),
            transaction_word_count
          )},
          {"subtitle_locator_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::subtitle_locator_begin),
            subtitle_locator_word_count
          )},
          {"subtitle_condition_words", gpu_trace_word_array(
            ring,
            base + word_index(record_word_e::subtitle_condition_begin),
            subtitle_condition_word_count
          )},
        });
      }
      return {
        {"schema", decoded_trace_schema},
        {"role", "decoded diagnostic history; no rendering authority"},
        {"ring", {
          {"schema", ring_schema}, {"tag", ring_tag}, {"capacity", capacity},
          {"record_words", record_word_count}, {"next_sequence", next_sequence},
          {"next_slot", next_slot}, {"committed_count", count},
          {"oldest_sequence", oldest_sequence},
        }},
        {"capture_match", {
          {"matched_frame_id", completed.matched_frame_id},
          {"analysis_generation", completed.depth_input_region.analysis_generation},
          {"source_width", completed.depth_input_region.width()},
          {"source_height", completed.depth_input_region.height()},
          {"field_width", completed.model_width},
          {"field_height", completed.model_height},
          {"sequence", matched_sequence},
        }},
        {"records", std::move(records)},
      };
    }

    struct gpu_trace_publication_t {
      bool available = false;
      std::string raw_sha256;
      std::string decoded_sha256;
      std::string contract_sha256;
      nlohmann::json summary = nullptr;
    };

    gpu_trace_publication_t publish_gpu_trace_artifacts(
      const captured_dump_job &job,
      const std::filesystem::path &directory
    ) {
      gpu_trace_publication_t result;
      if (job.gpu_trace_ring.empty() || !job.completed.gpu_trace_provenance ||
          !gpu_trace_shader_identity_matches_contract(
            job.completed.gpu_trace_provenance
          ) || !detail::gpu_trace_ring_is_canonical(
            job.gpu_trace_ring, job.completed
          )) {
        return result;
      }
      const auto raw_path = directory / "gpu_trace_ring.u32";
      const auto decoded_path = directory / "gpu_trace.json";
      const auto contract_path = directory / "gpu_trace_contract.json";
      const auto decoded = gpu_trace_decoded_document(
        job.gpu_trace_ring, job.completed
      );
      const auto contract = gpu_trace_contract_document(
        *job.completed.gpu_trace_provenance
      );
      const auto remove_optional_artifacts = [&]() {
        std::error_code ignored;
        std::filesystem::remove(raw_path, ignored);
        ignored.clear();
        std::filesystem::remove(decoded_path, ignored);
        ignored.clear();
        std::filesystem::remove(contract_path, ignored);
      };
      if (!write_bytes(
            raw_path, job.gpu_trace_ring.data(), job.gpu_trace_ring.size()
          ) || !write_json(decoded_path, decoded) ||
          !write_json(contract_path, contract)) {
        remove_optional_artifacts();
        return result;
      }
      result.raw_sha256 = models::file_sha256_hex(raw_path);
      result.decoded_sha256 = models::file_sha256_hex(decoded_path);
      result.contract_sha256 = models::file_sha256_hex(contract_path);
      if (result.raw_sha256.empty() || result.decoded_sha256.empty() ||
          result.contract_sha256.empty()) {
        remove_optional_artifacts();
        return {};
      }
      result.available = true;
      result.summary = {
        {"available", true},
        {"required", false},
        {"rendering_authority", false},
        {"raw_artifact", "gpu_trace_ring.u32"},
        {"decoded_artifact", "gpu_trace.json"},
        {"contract_artifact", "gpu_trace_contract.json"},
        {"record_count", decoded.at("ring").at("committed_count")},
        {"oldest_sequence", decoded.at("ring").at("oldest_sequence")},
        {"next_sequence", decoded.at("ring").at("next_sequence")},
        {"matched_sequence", decoded.at("capture_match").at("sequence")},
        {"source_closure_sha256",
          job.completed.gpu_trace_provenance->source_closure_sha256},
      };
      return result;
    }

    nlohmann::json window_region_document(
      const window_region_snapshot &region
    ) {
      std::ostringstream hwnd;
      hwnd << "0x" << std::hex << std::uppercase << region.hwnd;
      return {
        {"schema", window_region_schema},
        {"capture", "same matched source/color/depth/render frame as the parent Dump 3D package"},
        {"role", "matched-window region provenance; no independent geometry or renderer authority"},
        {"authority_kind", window_region_authority_kind_name(region.authority_kind)},
        {"matched_frame_id", region.matched_frame_id},
        {"coordinate_space", {
          {"name", "matched-source-pixels"},
          {"rect_semantics", "half-open [left, top, right, bottom)"},
          {"source_extent_px", {
            {"width", region.source_width},
            {"height", region.source_height},
          }},
          {"capture_rect_px", {
            {"left", region.left},
            {"top", region.top},
            {"right", region.right},
            {"bottom", region.bottom},
          }},
        }},
        {"identity", {
          {"hwnd", hwnd.str()},
          {"process_id", region.process_id},
          {"document_id", region.document_id},
          {"video_id", region.video_id},
          {"generation", region.generation},
        }},
        {"freshness", {
          {"latest_observation_age_ms_at_capture", region.latest_observation_age_ms_at_capture},
          {"maximum_observation_age_ms", region.maximum_observation_age_ms},
          {"geometry_continuity_ms_at_capture", region.geometry_continuity_ms_at_capture},
          {"source_content_age_ms_at_capture", region.source_content_age_ms_at_capture},
          {"fresh", true},
          {"causal_geometry", true},
        }},
      };
    }

    bool scalar_tensor_snapshot_matches(
      const texture_snapshot &snapshot,
      const std::uint32_t width,
      const std::uint32_t height
    ) noexcept {
      return snapshot.desc.Width == width && snapshot.desc.Height == height &&
             snapshot.desc.Format == DXGI_FORMAT_R32_FLOAT;
    }

    bool bgra_mask_snapshot_matches(
      const texture_snapshot &snapshot,
      const std::uint32_t width,
      const std::uint32_t height
    ) noexcept {
      return snapshot.desc.Width == width && snapshot.desc.Height == height &&
             (snapshot.desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
              snapshot.desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
    }

    nlohmann::json source_rect_document(
      const std::uint32_t left,
      const std::uint32_t top,
      const std::uint32_t right,
      const std::uint32_t bottom
    ) {
      return {
        {"left", left},
        {"top", top},
        {"right", right},
        {"bottom", bottom},
      };
    }

    std::string depth_input_region_error(const frame &completed) {
      const auto &region = completed.depth_input_region;
      if (!region.valid()) {
        return "invalid full-source/ROI analysis rectangle";
      }
      if (region.source_width == 0u || region.source_height == 0u ||
          completed.matched_frame_id == 0u) {
        return "missing matched-frame or source identity";
      }
      if (!region.video_region) {
        if (completed.depth_video_plan) {
          return "full-source completion carries an ROI planner result";
        }
        return {};
      }
      if (!completed.depth_video_plan || !completed.window_region) {
        return "ROI completion is missing its planner result or window-region provenance";
      }
      if (completed.window_region_observer_status != "ok" ||
          completed.window_region_mapping_status != "ok") {
        return "ROI completion is not backed by an accepted observer/mapping state";
      }
      const auto &provenance = *completed.window_region;
      const auto expected_authority =
        provenance.authority_kind == window_region_authority_kind_e::chromium_video ?
          models::depth_analysis_authority_e::chromium_video :
          models::depth_analysis_authority_e::foreground_client;
      if (region.authority != expected_authority) {
        return "ROI estimator authority kind does not match window-region provenance";
      }
      const auto provenance_validation = validate_window_region(
        provenance,
        completed.matched_frame_id,
        region.source_width,
        region.source_height
      );
      if (provenance_validation != window_region_error::none) {
        return std::string {"ROI window-region provenance failed validation: "} +
               window_region_error_name(provenance_validation);
      }

      const models::depth_source_rect_t semantic_rect {
          static_cast<std::uint32_t>(provenance.left),
          static_cast<std::uint32_t>(provenance.top),
          static_cast<std::uint32_t>(provenance.right),
          static_cast<std::uint32_t>(provenance.bottom),
      };
      const models::depth_tensor_shape_t tensor_shape {
        completed.model_width,
        completed.model_height,
      };
      const auto expected_plan = models::plan_host_sbs_v2_video_region(
        semantic_rect,
        region.source_width,
        region.source_height,
        tensor_shape
      );
      if (!expected_plan || *expected_plan != *completed.depth_video_plan) {
        return "ROI planner result is not the deterministic authenticated integer contain fit";
      }
      const auto &input_rect = completed.depth_video_plan->source_rect;
      if (region.left != input_rect.left || region.top != input_rect.top ||
          region.right != input_rect.right || region.bottom != input_rect.bottom) {
        return "ROI estimator domain does not match its planner input rectangle";
      }
      if (completed.depth_video_plan->tensor_shape != tensor_shape) {
        return "ROI planner tensor does not match the completed model/depth tensor";
      }
      if (!region.tensor_content.valid(tensor_shape) ||
          region.tensor_content != completed.depth_video_plan->tensor_content) {
        return "ROI tensor-content rectangle does not match its authenticated planner";
      }
      return {};
    }

    float video_region_vertical_slope_source_u_per_source_v(
      const models::depth_input_region_t &region
    ) {
      const float source_width = static_cast<float>(region.source_width);
      const float source_height = static_cast<float>(region.source_height);
      const float roi_left = static_cast<float>(region.left) / source_width;
      const float roi_top = static_cast<float>(region.top) / source_height;
      const float roi_right = static_cast<float>(region.right) / source_width;
      const float roi_bottom = static_cast<float>(region.bottom) / source_height;
      const float roi_width = roi_right - roi_left;
      const float roi_height = roi_bottom - roi_top;
      const float source_height_in_source_u =
        source_height / std::max(source_width, 1.0f);
      const float roi_pixel_aspect =
        (roi_width * source_width) /
        std::max(roi_height * source_height, 1.0e-6f);
      const float vertical_slope =
        models::depth_coordinate_v2::max_vertical_shear * roi_pixel_aspect *
        (static_cast<float>(region.tensor_content.height()) /
         std::max(static_cast<float>(region.tensor_content.width()), 1.0f));
      return vertical_slope * source_height_in_source_u;
    }

    nlohmann::json depth_input_region_document(const frame &completed) {
      const auto &region = completed.depth_input_region;
      const bool roi = region.video_region;
      nlohmann::json authorization = nullptr;
      if (roi) {
        const auto &provenance = *completed.window_region;
        std::ostringstream hwnd;
        hwnd << "0x" << std::hex << std::uppercase << provenance.hwnd;
        authorization = {
          {"authority_kind", window_region_authority_kind_name(provenance.authority_kind)},
          {"observer_generation", provenance.generation},
          {"hwnd", hwnd.str()},
          {"process_id", provenance.process_id},
          {"document_id", provenance.document_id},
          {"video_id", provenance.video_id},
        };
      }
      const nlohmann::json semantic_rect = roi ?
        source_rect_document(
          static_cast<std::uint32_t>(completed.window_region->left),
          static_cast<std::uint32_t>(completed.window_region->top),
          static_cast<std::uint32_t>(completed.window_region->right),
          static_cast<std::uint32_t>(completed.window_region->bottom)
        ) :
        nlohmann::json(nullptr);
      const nlohmann::json outside = roi ?
        nlohmann::json {
          {"construction", "signed soft-threshold collar"},
          {"horizontal_slope_source_u_per_source_u",
           models::depth_coordinate_v2::max_horizontal_slope},
          {"vertical_slope_source_u_per_source_v",
           video_region_vertical_slope_source_u_per_source_v(region)},
          {"beyond_collar", "exact zero parallax"},
        } :
        nlohmann::json(nullptr);
      // Match the actual b2/HLSL path bit-for-bit in operation order. The live shader subtracts
      // two normalized float edges; serializing integer-width/source-width as a double would
      // describe a subtly different embedding for most nonzero-left rectangles.
      const float field_to_source_scale = roi ?
        static_cast<float>(region.right) / static_cast<float>(region.source_width) -
          static_cast<float>(region.left) / static_cast<float>(region.source_width) :
        1.0f;
      return {
        {"schema", 3},
        {"capture", "same matched source/color/model/depth/render frame as the parent Dump 3D package"},
        {"role", "authoritative analysis-domain placement and live-render embedding contract"},
        {"matched_frame_id", completed.matched_frame_id},
        {"mode", roi ? "window-region" : "full-source"},
        {"authorization", authorization},
        {"coordinate_space", {
          {"name", "matched-source-pixels"},
          {"rect_semantics", "half-open [left, top, right, bottom)"},
          {"source_extent_px", {
            {"width", region.source_width},
            {"height", region.source_height},
          }},
          {"semantic_rect_px", semantic_rect},
          {"inference_rect_px", source_rect_document(
            region.left,
            region.top,
            region.right,
            region.bottom
          )},
        }},
        {"analysis", {
          {"analysis_generation", region.analysis_generation},
          {"input_domain_reset", completed.input_domain_reset},
          {"tensor_extent_px", {
            {"width", completed.model_width},
            {"height", completed.model_height},
          }},
          {"tensor_content_rect_px", source_rect_document(
            region.tensor_content.left,
            region.tensor_content.top,
            region.tensor_content.right,
            region.tensor_content.bottom
          )},
          {"padded_area_fraction",
           roi ? completed.depth_video_plan->padded_area_fraction : 0.0f},
          {"fit_method", roi ?
            "centered-integer-contain-with-edge-replicated-excluded-padding" :
            "full-tensor"},
          {"crop_method", roi ?
            "same-format D3D11 CopySubresourceRegion" : "full-source"},
          {"scene_analysis_domain", roi ?
            "inference-rectangle-only" : "full-source"},
        }},
        {"renderer", {
          {"final_parallax_units", roi ?
            "roi-local-source-u" : "full-source-u"},
          {"full_source_parallax_scale", field_to_source_scale},
          {"inside_inference_rect", "no taper"},
          {"outside", outside},
          {"source_sampling", "full matched source; never clamp to inference rectangle"},
          {"inverse_iterations", 11},
        }},
      };
    }

    std::string timestamp_string() {
      char text[32] = "unknown";
      const std::time_t now = std::time(nullptr);
      std::tm local {};
      if (localtime_s(&local, &now) == 0) {
        std::strftime(text, sizeof(text), "%Y%m%d_%H%M%S", &local);
      }
      return text;
    }

    struct output_paths {
      std::filesystem::path temporary;
      std::filesystem::path final;
    };

    bool make_output_paths(
      const std::filesystem::path &root,
      output_paths &paths,
      std::error_code &error
    ) {
      static std::atomic<std::uint64_t> sequence {0};
      const auto ticks = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
      );
      for (unsigned attempt = 0; attempt < 128u; ++attempt) {
        const std::uint64_t serial =
          sequence.fetch_add(1u, std::memory_order_relaxed);
        std::ostringstream suffix;
        suffix << timestamp_string() << '_' << GetCurrentProcessId() << '_'
               << std::hex << ticks << '_' << serial;
        paths.final = root / ("dump_" + suffix.str());
        paths.temporary = root / (".partial_" + suffix.str());
        error.clear();
        if (
          !std::filesystem::exists(paths.final, error) && !error &&
          std::filesystem::create_directory(paths.temporary, error) && !error
        ) {
          return true;
        }
        if (error) {
          return false;
        }
      }
      return false;
    }

  }  // namespace

  namespace detail {

    std::string gpu_trace_contract_json(
      const models::host_sbs_gpu_trace_provenance_t &provenance
    ) {
      return gpu_trace_contract_document(provenance).dump();
    }

    std::string gpu_trace_decoded_json(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      return gpu_trace_decoded_document(ring, completed).dump();
    }

  }  // namespace detail

  dumper::dumper():
      async_(detail::publication_state::create()) {
    if (const char *override_dir = std::getenv("APOLLO_SBS_DUMP"); override_dir && *override_dir) {
      dir_ = override_dir;
      file_trigger_enabled_ = config::sunshine.diagnostics_enabled;
    } else if (!config::sunshine.log_file.empty()) {
      dir_ =
        std::filesystem::path(config::sunshine.log_file).parent_path() /
        "sbs_dump";
    } else {
      dir_ = "sbs_dump";
    }
  }

  dumper::~dumper() {
    // Invalidate retry callbacks first, then release the session handle. Accepted publication
    // owns only CPU snapshots plus shared state and finishes on the process-lifetime queue.
    cancel_pending_request();
    async_.reset();
  }

  void dumper::set_button_request(std::shared_ptr<std::atomic<bool>> request) {
    button_request_ = std::move(request);
    if (async_) {
      async_->allow_retries_and_token();
    }
  }

  void dumper::cancel_pending_request() noexcept {
    if (async_) {
      async_->cancel_retries(button_request_);
    }
    const bool remove_file_trigger =
      file_trigger_enabled_ || file_trigger_pending_;
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    pending_gpu_capture_.reset();
    retry_not_before_ = {};
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    file_trigger_pending_ = false;
    file_trigger_enabled_ = false;
    if (!remove_file_trigger) {
      return;
    }
    try {
      std::error_code error;
      std::filesystem::remove(dir_ / "dump.trigger", error);
    } catch (...) {
      // Permanent estimator failure already makes this request impossible. Cleanup is best-effort
      // and must never turn the encode-thread failure handling into another exception path.
    }
  }

  void dumper::reject_pending_request() noexcept {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    pending_gpu_capture_.reset();
    retry_not_before_ = {};
    if (auto *button = button_request_.get()) {
      button->store(false, std::memory_order_relaxed);
    }
    if (file_trigger_pending_) {
      try {
        std::error_code error;
        std::filesystem::remove(dir_ / "dump.trigger", error);
      } catch (...) {
      }
    }
    file_trigger_pending_ = false;
  }

  bool dumper::snapshot_requested() {
    snapshot_armed_for_dump_ = false;
    prepared_frame_id_ = 0;
    if (async_ && async_->take_file_retry_pending()) {
      file_trigger_pending_ = true;
    }
    if (async_ && async_->take_trigger_remove_failed()) {
      file_trigger_enabled_ = false;
    }
    if (async_ && async_->take_publication_failed()) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
    }
    // The button latch remains armed while publication is active, so a later click is retained
    // for the next frame instead of replacing or aliasing the in-flight package.
    if (pending_gpu_capture_ || (async_ && async_->busy())) {
      return false;
    }
    if (std::chrono::steady_clock::now() < retry_not_before_) {
      return false;
    }
    const auto *button = button_request_.get();
    if (button && button->load(std::memory_order_relaxed)) {
      if (async_) {
        async_->allow_retries_and_token();
      }
      snapshot_armed_for_dump_ = true;
      return true;
    }
    if (!file_trigger_enabled_) {
      return false;
    }
    if (file_trigger_pending_) {
      if (async_) {
        async_->allow_retries_and_token();
      }
      snapshot_armed_for_dump_ = true;
      return true;
    }
    (void) needs_conversion_poll();
    snapshot_armed_for_dump_ = file_trigger_pending_;
    if (file_trigger_pending_ && async_) {
      async_->allow_retries_and_token();
    }
    return file_trigger_pending_;
  }

  bool dumper::needs_conversion_poll() const noexcept {
    if (pending_gpu_capture_) {
      return true;
    }
    if (async_ && async_->busy()) {
      return false;
    }
    if (std::chrono::steady_clock::now() < retry_not_before_) {
      return false;
    }
    const auto *button = button_request_.get();
    if (button && button->load(std::memory_order_acquire)) {
      return true;
    }
    if (!file_trigger_enabled_) {
      return false;
    }
    if (file_trigger_pending_) {
      return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now < next_file_trigger_poll_) {
      return false;
    }
    next_file_trigger_poll_ = now + std::chrono::seconds(1);
    try {
      std::error_code error;
      file_trigger_pending_ =
        std::filesystem::exists(dir_ / "dump.trigger", error) && !error;
    } catch (...) {
      file_trigger_pending_ = false;
    }
    return file_trigger_pending_;
  }

  bool dumper::prepare_requested_v2_frame(
    const std::uint64_t matched_frame_id
  ) noexcept {
    auto *button = button_request_.get();
    const bool requested =
      (button && button->load(std::memory_order_relaxed)) ||
      file_trigger_pending_;
    if (!requested || !snapshot_armed_for_dump_ || pending_gpu_capture_ ||
        matched_frame_id == 0) {
      return false;
    }
    // Do not read shadow_state here. The subsequent staging batch captures that buffer and the
    // packed SBS in one command-stream transaction, then validates the camera from those exact
    // bytes once the terminal event query reports completion.
    prepared_frame_id_ = matched_frame_id;
    return true;
  }

  namespace {

  dump_publish_result publish_captured_dump(const captured_dump_job &job) {
    dump_publish_result result;
    const auto &completed = job.completed;
    const bool video_region = completed.depth_input_region.video_region;
    const auto &cfg = job.cfg;
    const auto &capture_preprocess = *job.preprocess;
    const bool hdr =
      completed.color_space == models::input_color_space::scrgb_hdr;
    const bool by_button = job.by_button;
    const bool by_file = job.by_file;
    const auto &model_identity = *completed.raw_model_provenance;

    const std::filesystem::path &trigger = job.trigger;
    std::error_code error;
    output_paths paths;
    bool success = false;
    try {
      const auto &normalization = job.normalization;
      const bool scene_cut_bridge_state_available =
        job.scene_cut_bridge_state_available;

      if (
        !std::filesystem::create_directories(job.root, error) && error
      ) {
        BOOST_LOG(warning) << "SBS debug dump: cannot create root "sv
                           << job.root.string() << ": " << error.message();
        result.error = error.message();
        return result;
      }
      if (!make_output_paths(job.root, paths, error)) {
        BOOST_LOG(warning) << "SBS debug dump: cannot reserve a unique output folder in "sv
                           << job.root.string()
                           << (error ? ": " + error.message() : "."s);
        result.error = error ? error.message() : "cannot reserve output folder";
        return result;
      }

      do {
        const auto &source = job.source;
        const auto &depth_input_source = completed.depth_input_region.video_region ?
                                           job.depth_input_source :
                                           job.source;
        const auto &warp_depth = job.warp_depth;
        const auto &sbs = job.sbs;
        const bool subtitle_slr13_active =
          !job.subtitle_ocr_record.empty() &&
          !job.subtitle_locator_state.empty();
        const nlohmann::json subtitle_conditioning = subtitle_slr13_active ?
          nlohmann::json {
            {"mode", "subtitle-slr13"},
            {"request", true},
            {"producer", subtitle_ocr_producer_contract_json(
              *completed.parallax_v2_shader_provenance
            )},
            {"resolver", subtitle_locator_resolver_contract_json(
              *completed.parallax_v2_shader_provenance
            )},
            {"artifacts", {
              {"ocr_record", "subtitle_ocr_record.u32"},
              {"locator_state", "subtitle_locator_state.u32"},
              {"base_field", "shadow_base_final_parallax.f32"},
              {"conditioned_field", "shadow_final_parallax.f32"},
            }},
          } :
          nlohmann::json {
            {"mode", "none"},
            {"request", nullptr},
            {"producer", nullptr},
            {"resolver", nullptr},
            {"artifacts", nlohmann::json::object()},
          };
        const auto region_error = depth_input_region_error(completed);
        if (!region_error.empty()) {
          BOOST_LOG(warning) << "SBS debug dump: depth input region rejected: "sv
                             << region_error << '.';
          break;
        }
        if (
          source.desc.Width != completed.depth_input_region.source_width ||
          source.desc.Height != completed.depth_input_region.source_height ||
          depth_input_source.desc.Width != completed.depth_input_region.width() ||
          depth_input_source.desc.Height != completed.depth_input_region.height() ||
          depth_input_source.desc.Format != source.desc.Format
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: full-source or analysis-source texture extent/format does "sv
               "not match the authenticated depth input region."sv;
          break;
        }
        const auto tensor_width = static_cast<std::uint32_t>(completed.model_width);
        const auto tensor_height = static_cast<std::uint32_t>(completed.model_height);
        if (
          !scalar_tensor_snapshot_matches(warp_depth, tensor_width, tensor_height) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_coordinate,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_candidate,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_ownership_refined,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_vertical,
            tensor_width,
            tensor_height
          ) ||
          !scalar_tensor_snapshot_matches(
            job.shadow_vertical_conditioned,
            tensor_width,
            tensor_height
          ) ||
          (subtitle_slr13_active && !scalar_tensor_snapshot_matches(
            job.shadow_base_final,
            tensor_width,
            tensor_height
          )) ||
          !scalar_tensor_snapshot_matches(job.shadow_final, tensor_width, tensor_height)
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: a completed parallax-v2 tensor texture does not match "sv
               "the authenticated model extent/R32_FLOAT contract."sv;
          break;
        }
        const auto exact_scalar_snapshot = [](
          const texture_snapshot &left,
          const texture_snapshot &right
        ) {
          return left.desc.Width == right.desc.Width &&
                 left.desc.Height == right.desc.Height &&
                 left.desc.Format == right.desc.Format &&
                 left.row_bytes == right.row_bytes && left.bytes == right.bytes;
        };
        if (!exact_scalar_snapshot(warp_depth, job.shadow_final)) {
          BOOST_LOG(warning)
            << "SBS debug dump: warp_depth is not the exact displayed parallax field."sv;
          break;
        }
        if (sbs.desc.Width < 2u || (sbs.desc.Width & 1u) != 0u) {
          BOOST_LOG(warning)
            << "SBS debug dump: packed output width is not an even two-eye extent."sv;
          break;
        }
        if (
          !write_color_preview(
            paths.temporary / "source.png",
            source,
            completed.color_space
          ) ||
          !write_color_preview(
            paths.temporary / "depth_input_source.png",
            depth_input_source,
            completed.color_space
          ) ||
          !write_json(
            paths.temporary / "depth_input_region.json",
            depth_input_region_document(completed)
          ) ||
          !write_json(
            paths.temporary / "subtitle_conditioning.json",
            subtitle_conditioning
          ) ||
          (subtitle_slr13_active && (
            !write_bytes(
              paths.temporary / "subtitle_ocr_record.u32",
              job.subtitle_ocr_record.data(),
              job.subtitle_ocr_record.size()
            ) ||
            !write_bytes(
              paths.temporary / "subtitle_locator_state.u32",
              job.subtitle_locator_state.data(),
              job.subtitle_locator_state.size()
            )
          )) ||
          !dump_model_input(
            job.model_input,
            completed.model_width,
            completed.model_height,
            paths.temporary,
            capture_preprocess
          )
        ) {
          break;
        }

        raw_depth_dump_stats raw_stats;
        if (
          !dump_raw_depth(
            job.raw_depth,
            completed.raw_width,
            completed.raw_height,
            paths.temporary,
            raw_stats
          ) ||
          !write_float_texture_artifacts(
            paths.temporary / "warp_depth.f32",
            paths.temporary / "warp_depth_shape.json",
            warp_depth,
            video_region ?
              "exact one-eye ROI-local source-U from the orientation-selective vertical conditioner followed by the row majorant; renderer authority requires depth_input_region embedding" :
              "exact one-eye full-source-U from the orientation-selective vertical conditioner followed by the row majorant sampled by live Host-SBS V2 reprojection"
          ) ||
          !write_scalar_previews(
            paths.temporary / "warp_depth.png",
            paths.temporary / "warp_depth_heat.png",
            warp_depth
          ) ||
          !write_color_preview(
            paths.temporary / "sbs.png",
            sbs,
            completed.color_space
          )
        ) {
          break;
        }

        nlohmann::json adaptive = nullptr;
        bool adaptive_available = false;
        if (scene_cut_bridge_state_available) {
          try {
            adaptive_available = dump_adaptive_state(
              job.adaptive_state,
              normalization,
              completed,
              paths.temporary,
              adaptive
            );
          } catch (...) {
            // Scene-cut bridge evidence is optional and has no live V2 geometry authority.
          }
        }
        // The completion ring is optional diagnostic evidence. Missing resources, a torn ring,
        // provenance mismatch, or publication failure omits only these three files; it must never
        // reject the authenticated core Dump 3D package.
        const auto gpu_trace = publish_gpu_trace_artifacts(
          job, paths.temporary
        );
        const bool held_subtitle_evidence = subtitle_slr13_active &&
          subtitle_u64(job.subtitle_locator_state, 22u) !=
            completed.matched_frame_id;
        if (subtitle_slr13_active && held_subtitle_evidence &&
            !gpu_trace.available) {
          BOOST_LOG(warning)
            << "SBS debug dump: held OCR8/SLR13 evidence requires its authenticated "sv
               "GPU trace artifact; publication aborted."sv;
          break;
        }
        // Bind provenance to the exact bytes that were successfully written into this
        // transaction directory. Hashing the files (rather than reconstructed vectors) also
        // covers byte order and the canonical JSON serialization of the input-shape contract.
        const std::string raw_depth_sha256 = models::file_sha256_hex(
          paths.temporary / "raw_depth.f32"
        );
        const std::string model_input_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input.f32"
        );
        const std::string model_input_shape_sha256 = models::file_sha256_hex(
          paths.temporary / "model_input_shape.json"
        );
        const std::string depth_input_region_sha256 = models::file_sha256_hex(
          paths.temporary / "depth_input_region.json"
        );
        const std::string subtitle_conditioning_sha256 = models::file_sha256_hex(
          paths.temporary / "subtitle_conditioning.json"
        );
        const std::string subtitle_ocr_record_sha256 = subtitle_slr13_active ?
          models::file_sha256_hex(paths.temporary / "subtitle_ocr_record.u32") :
          std::string {};
        const std::string subtitle_locator_state_sha256 = subtitle_slr13_active ?
          models::file_sha256_hex(paths.temporary / "subtitle_locator_state.u32") :
          std::string {};
        if (raw_depth_sha256.empty() || model_input_sha256.empty() ||
            model_input_shape_sha256.empty() || depth_input_region_sha256.empty() ||
            subtitle_conditioning_sha256.empty() ||
            (subtitle_slr13_active && (
              subtitle_ocr_record_sha256.empty() ||
              subtitle_locator_state_sha256.empty()
            ))) {
          BOOST_LOG(warning)
            << "SBS debug dump: cannot hash the exact raw-model/domain artifacts; "sv
               "publication aborted."sv;
          break;
        }

        const auto &shadow_coordinate = job.shadow_coordinate;
        const auto &shadow_candidate = job.shadow_candidate;
        const auto &shadow_ownership_refined = job.shadow_ownership_refined;
        const auto &shadow_vertical = job.shadow_vertical;
        const auto &shadow_vertical_conditioned = job.shadow_vertical_conditioned;
        const auto &shadow_base_final = job.shadow_base_final;
        const auto &shadow_final = job.shadow_final;
        nlohmann::json shadow_summary = nullptr;
        if (
          !dump_shadow_float_texture(
             shadow_coordinate,
             paths.temporary,
             "shadow_coordinate",
             "parallax-v2 canonical unbounded coordinate u; diagnostic only"
           ) ||
           !dump_shadow_float_texture(
             shadow_candidate,
             paths.temporary,
             "shadow_candidate_parallax",
             video_region ?
               "parallax-v2 immutable signed ROI-local source-U pre-conditioner geometry evidence; never renderer authority without depth_input_region embedding" :
               "parallax-v2 immutable signed full-source-U pre-conditioner geometry evidence; never geometry authority"
           ) ||
           !dump_shadow_float_texture(
             shadow_ownership_refined,
             paths.temporary,
             "shadow_ownership_refined_parallax",
             video_region ?
               "parallax-v2 signed ROI-local source-U candidate after conservative full-resolution crop-contour foreground ownership; consumed by the vertical conditioner" :
               "parallax-v2 signed full-source-U candidate after conservative full-resolution source-contour foreground ownership; consumed by the vertical conditioner"
           ) ||
           !dump_shadow_float_texture(
             shadow_vertical,
             paths.temporary,
             "shadow_vertical_majorant",
             video_region ?
               "parallax-v2 ROI-local source-U least column-wise upper envelope of the ownership-refined candidate; diagnostic evidence only" :
               "parallax-v2 full-source-U least column-wise upper envelope of the ownership-refined candidate; diagnostic evidence only"
           ) ||
           !dump_shadow_float_texture(
             shadow_vertical_conditioned,
             paths.temporary,
             "shadow_vertical_conditioned",
             video_region ?
               "parallax-v2 ROI-local source-U fixed 75/25 share of the column upper/lower envelopes; neutral intermediate consumed by the row majorant" :
               "parallax-v2 full-source-U fixed 75/25 share of the column upper/lower envelopes; neutral intermediate consumed by the row majorant"
           ) ||
           (subtitle_slr13_active && !dump_shadow_float_texture(
             shadow_base_final,
             paths.temporary,
             "shadow_base_final_parallax",
             video_region ?
               "ordinary parallax-v2 crop-local source-U field after the horizontal majorant and before SLR13; renderer authority requires current SLR13 conditioning plus depth_input_region embedding" :
               "ordinary parallax-v2 full-source-U field after the horizontal majorant and before SLR13 conditioning"
           )) ||
           !dump_shadow_float_texture(
             shadow_final,
             paths.temporary,
             "shadow_final_parallax",
             video_region ?
               "complete atomic crop-local source-U field after SLR13 conditioning; sampled directly with depth_input_region embedding" :
               "complete atomic full-source-U field after SLR13 conditioning; sole live V2 render position authority"
           ) ||
           !dump_parallax_v2_state(
             completed,
             job.shadow_state,
             job.shadow_frame_stats,
             paths.temporary,
             shadow_summary
           )
        ) {
          break;
        }

        const bool warp_map_available = job.warp_map_available;
        const bool warp_mask_available = job.warp_mask_available;
        warp_map_dump_stats warp_map_stats;
        const auto &warp_map = job.warp_map;
        const auto &warp_mask = job.warp_mask;
        if (
          (warp_map_available &&
           !scalar_tensor_snapshot_matches(
             warp_map,
             sbs.desc.Width,
             sbs.desc.Height
           )) ||
          (warp_mask_available &&
           !bgra_mask_snapshot_matches(
             warp_mask,
             sbs.desc.Width,
             sbs.desc.Height
           ))
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: dump-only map/mask resources do not match the packed "sv
               "output extent or format; publication aborted."sv;
          break;
        }
        if (
          warp_map_available &&
          (!dump_warp_map(
             warp_map,
             source.desc.Width,
             source.desc.Height,
             video_region,
             paths.temporary,
             warp_map_stats
           ))
        ) {
          break;
        }
        if (
          warp_mask_available &&
          (!write_color_preview(
             paths.temporary / "warp_mask.png",
             warp_mask,
             models::input_color_space::srgb
           ))
        ) {
          break;
        }
        std::string warp_map_sha256;
        if (completed.depth_input_region.video_region) {
          if (!warp_map_available || !warp_mask_available) {
            BOOST_LOG(warning)
              << "SBS debug dump: ROI publication requires a full-source inverse map and "sv
                 "boundary mask; publication aborted."sv;
            break;
          }
          warp_map_sha256 = models::file_sha256_hex(
            paths.temporary / "warp_map.f32"
          );
          if (warp_map_sha256.empty()) {
            BOOST_LOG(warning)
              << "SBS debug dump: ROI full-source inverse map could not be hashed; "sv
                 "publication aborted."sv;
            break;
          }
        }

        if (
          sbs.desc.Width < 2u || (sbs.desc.Width & 1u) != 0u ||
          source.desc.Width == 0u || source.desc.Height == 0u ||
          sbs.desc.Height == 0u
        ) {
          break;
        }
        bool window_region_available = false;
        std::string window_region_sha256;
        if (completed.window_region) {
          const auto validation = validate_window_region(
            *completed.window_region,
            completed.matched_frame_id,
            source.desc.Width,
            source.desc.Height
          );
          if (validation == window_region_error::none) {
            window_region_available = write_json(
              paths.temporary / "window_region.json",
              window_region_document(*completed.window_region)
            );
            if (!window_region_available) {
              BOOST_LOG(warning)
                << "SBS debug dump: optional matched-frame window-region provenance could not be written; continuing without it."sv;
            } else if (completed.depth_input_region.video_region) {
              window_region_sha256 = models::file_sha256_hex(
                paths.temporary / "window_region.json"
              );
            }
          } else {
            BOOST_LOG(warning)
              << "SBS debug dump: optional window-region provenance rejected ("sv
              << window_region_error_name(validation)
              << "); continuing without it."sv;
          }
        }
        if (
          completed.depth_input_region.video_region &&
          (!window_region_available || window_region_sha256.empty())
        ) {
          BOOST_LOG(warning)
            << "SBS debug dump: ROI publication requires exact matched-frame window-region "sv
               "provenance; publication aborted."sv;
          break;
        }
        const std::uint32_t eye_width = sbs.desc.Width / 2u;
        const std::uint32_t eye_height = sbs.desc.Height;
        const float source_aspect =
          static_cast<float>(source.desc.Width) /
          static_cast<float>(source.desc.Height);
        const float eye_aspect =
          static_cast<float>(eye_width) / static_cast<float>(eye_height);
        const float content_scale_x =
          eye_aspect > source_aspect ? source_aspect / eye_aspect : 1.0f;
        const float content_scale_y =
          eye_aspect < source_aspect ? eye_aspect / source_aspect : 1.0f;

        const std::string warp_scalar_stage = video_region ?
          "crop-local final parallax field embedded by live V2 reprojection" :
          "atomic final parallax field sampled by live V2 reprojection";
        const std::string warp_scalar_description = video_region ?
          "Exact complete atomic one-eye ROI-local source-U field. Renderer authority is this field together with depth_input_region.json, which embeds it into the full source and supplies the outside-only zero-plane collar." :
          "Exact complete atomic one-eye full-source-U field sampled by the live V2 11-step contractive inverse.";
        // Bind every V2 geometry field to the exact bytes written into this transaction
        // directory. Metadata-only descriptors let a truncated or internally inconsistent
        // geometry dump validate cleanly, which silently poisons every downstream offline
        // investigation that trusts validated dumps.
        const std::string warp_depth_sha256 =
          models::file_sha256_hex(paths.temporary / "warp_depth.f32");
        const std::string shadow_coordinate_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_coordinate.f32");
        const std::string shadow_candidate_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_candidate_parallax.f32");
        const std::string shadow_ownership_sha256 = models::file_sha256_hex(
          paths.temporary / "shadow_ownership_refined_parallax.f32"
        );
        const std::string shadow_vertical_majorant_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_vertical_majorant.f32");
        const std::string shadow_vertical_conditioned_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_vertical_conditioned.f32");
        const std::string shadow_base_final_sha256 = subtitle_slr13_active ?
          models::file_sha256_hex(
            paths.temporary / "shadow_base_final_parallax.f32"
          ) :
          std::string {};
        const std::string shadow_final_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_final_parallax.f32");
        const std::string shadow_state_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_state.json");
        const std::string shadow_frame_stats_sha256 =
          models::file_sha256_hex(paths.temporary / "shadow_frame_stats.json");
        if (warp_depth_sha256.empty() || shadow_coordinate_sha256.empty() ||
            shadow_candidate_sha256.empty() || shadow_ownership_sha256.empty() ||
            shadow_vertical_majorant_sha256.empty() ||
            shadow_vertical_conditioned_sha256.empty() || shadow_final_sha256.empty() ||
            shadow_state_sha256.empty() || shadow_frame_stats_sha256.empty() ||
            (subtitle_slr13_active && shadow_base_final_sha256.empty())) {
          BOOST_LOG(warning)
            << "SBS debug dump: failed to hash a written V2 geometry field; dump rejected."sv;
          break;
        }
        nlohmann::json artifacts = nlohmann::json::object();
        artifacts["source.png"] = artifact_description(
          true,
          true,
          "captured color supplied to Host SBS",
          "Preview decoded according to the matched frame's declared transfer: direct sRGB, linear-SDR OETF, or HDR tone map plus OETF."
        );
        artifacts["depth_input_source.png"] = artifact_description(
          true,
          false,
          "model-depth input source preview",
          "Spatially exact full-source or cropped color input submitted to the calibrated preprocess; transfer-aware PNG is diagnostic only and never numeric model authority."
        );
        artifacts["depth_input_region.json"] = hashed_artifact_description(
          true,
          true,
          "depth analysis input region",
          "Authoritative full-source placement, crop-local analysis domain, and live-render embedding for every model/depth/parallax artifact in this matched package.",
          depth_input_region_sha256
        );
        artifacts["subtitle_conditioning.json"] = hashed_artifact_description(
          true,
          true,
          "subtitle conditioning authority",
          "Canonical current-schema subtitle authority descriptor: none or exact OCR8/SLR13.",
          subtitle_conditioning_sha256
        );
        if (subtitle_slr13_active) {
          artifacts["subtitle_ocr_record.u32"] = hashed_artifact_description(
            true,
            true,
            "OCR8 subtitle boxes for atomic target",
            "Exact 208-word little-endian OCR8 record bound to the atomic target's publication frame (current authenticated observation or held ordinary reuse), analysis generation, source, field, and ROI.",
            subtitle_ocr_record_sha256
          );
          artifacts["subtitle_locator_state.u32"] = hashed_artifact_description(
            true,
            true,
            "compact SLR13 subtitle authority state",
            "Exact 80-word little-endian SLR13 owner, pending, stabilized local-plane target, bounded unreliable-measurement hold count, and target-publication authority state.",
            subtitle_locator_state_sha256
          );
        }
        artifacts["gpu_trace_ring.u32"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "diagnostic GPU accepted-root completion history",
            "Raw little-endian 300-slot completion ring. This is diagnostic evidence only and has no rendering authority.",
            gpu_trace.raw_sha256
          ) :
          artifact_description(
            false,
            false,
            "diagnostic GPU accepted-root completion history",
            "Optional diagnostic ring was unavailable, untrusted, torn, or could not be published; the core dump remains complete."
          );
        artifacts["gpu_trace.json"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "decoded diagnostic GPU history",
            "Chronological decoded records with raw transaction, SLR13, and condition words plus authenticated device dispositions.",
            gpu_trace.decoded_sha256
          ) :
          artifact_description(
            false,
            false,
            "decoded diagnostic GPU history",
            "Optional decoded trace is absent with the raw diagnostic ring."
          );
        artifacts["gpu_trace_contract.json"] = gpu_trace.available ?
          hashed_artifact_description(
            true,
            false,
            "diagnostic GPU trace wire contract",
            "Exact ring offsets, receipt ABI, enum values, embedded state schemas, and authenticated shader source closure.",
            gpu_trace.contract_sha256
          ) :
          artifact_description(
            false,
            false,
            "diagnostic GPU trace wire contract",
            "Optional trace contract is absent with the raw diagnostic ring."
          );
        artifacts["model_input.f32"] = artifact_description(
          true,
          true,
          "exact neural-network input",
          "Float32-le NCHW tensor after preprocessing and ImageNet normalization."
        );
        artifacts["model_input.png"] = artifact_description(
          true,
          true,
          "neural-network input preview",
          "ImageNet mean/std reversed without a second sRGB transfer function."
        );
        artifacts["model_input_shape.json"] = artifact_description(
          true,
          true,
          "model-input contract",
          "Dimensions, layout, normalization, and preview semantics."
        );
        artifacts["raw_depth.f32"] = artifact_description(
          true,
          true,
          "exact model output",
          "Float32-le raw depth before normalization or temporal processing."
        );
        artifacts["raw_depth.png"] = artifact_description(
          true,
          true,
          "raw model output preview",
          "Finite p2-p98 grayscale preview; not the tensor's numeric contract."
        );
        artifacts["raw_depth_heat.png"] = artifact_description(
          true,
          true,
          "raw model output preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["raw_shape.json"] = artifact_description(
          true,
          true,
          "raw-depth contract",
          "Dimensions, scalar statistics, and preview bounds."
        );
        artifacts["warp_depth.png"] = artifact_description(
          true,
          true,
          warp_scalar_stage,
          "Grayscale preview of the exact orientation-selective conditioned field sampled by the warp."
        );
        artifacts["warp_depth.f32"] = hashed_artifact_description(
          true,
          true,
          warp_scalar_stage,
          warp_scalar_description,
          warp_depth_sha256
        );
        artifacts["warp_depth_shape.json"] = artifact_description(
          true,
          true,
          "actual displayed parallax-field contract",
          "Dimensions, layout, units, and scalar range for warp_depth.f32."
        );
        artifacts["warp_depth_heat.png"] = artifact_description(
          true,
          true,
          warp_scalar_stage,
          "Jet preview of the exact orientation-selective conditioned field sampled by the warp."
        );
        artifacts["adaptive_state.json"] = artifact_description(
          adaptive_available,
          false,
          "scene-cut bridge comparison state",
          adaptive_available ?
            "Comparison-only cut flags, counters, and normalization state; no live V2 geometry authority." :
            "Unavailable; scene-cut bridge evidence never gates an authenticated live V2 dump."
        );
        artifacts["window_region.json"] = video_region ?
          hashed_artifact_description(
            true,
            true,
            "matched-frame window region provenance",
            "Validated semantic rectangle, authority kind, stable identity, freshness, and causal provenance from which the authoritative ROI input region was planned; not renderer authority by itself.",
            window_region_sha256
          ) :
          artifact_description(
            window_region_available,
            false,
            "matched-frame window region provenance",
            "Validated half-open capture rectangle, source extent, authority kind, identity, and freshness; provenance only and never independent renderer authority."
          );
        artifacts["shadow_coordinate.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 canonical coordinate diagnostic",
          "Exact float32-le unbounded canonical coordinate u; diagnostic only and never used by the live renderer.",
          shadow_coordinate_sha256
        );
        artifacts["shadow_coordinate_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate contract",
          "Dimensions, units, and finite scalar range."
        );
        artifacts["shadow_coordinate.png"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate preview",
          "Finite p2-p98 grayscale preview; not the numeric contract."
        );
        artifacts["shadow_coordinate_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 canonical coordinate preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["shadow_candidate_parallax.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 pre-limiter candidate displacement",
          video_region ?
            "Exact immutable signed one-eye ROI-local source-U before the spatial limiter; geometry evidence only and never renderer authority without depth_input_region embedding." :
            "Exact immutable signed one-eye full-source-U before the spatial limiter; geometry evidence only, never live render authority.",
          shadow_candidate_sha256
        );
        artifacts["shadow_candidate_parallax_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement contract",
          "Dimensions, units, and finite scalar range."
        );
        artifacts["shadow_candidate_parallax.png"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement preview",
          "Finite p2-p98 grayscale preview."
        );
        artifacts["shadow_candidate_parallax_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 pre-limiter candidate displacement preview",
          "Finite p2-p98 jet preview."
        );
        artifacts["shadow_ownership_refined_parallax.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 full-resolution contour ownership refinement",
          video_region ?
            "Exact signed one-eye ROI-local source-U after conservative full-resolution crop-local source-contour foreground ownership and before the vertical conditioner. The pass may only raise an authenticated candidate at a uniquely owned far-side boundary texel." :
            "Exact signed one-eye full-source-U after conservative full-resolution source-contour foreground ownership and before the vertical conditioner. The pass may only raise an authenticated candidate at a uniquely owned far-side boundary texel.",
          shadow_ownership_sha256
        );
        artifacts["shadow_ownership_refined_parallax_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 full-resolution contour ownership refinement contract",
          video_region ?
            "Crop-local dimensions, ROI-local source-U units, and finite scalar range for the ownership-refined candidate consumed by the vertical conditioner." :
            "Full-source dimensions, full-source-U units, and finite scalar range for the ownership-refined candidate consumed by the vertical conditioner."
        );
        artifacts["shadow_ownership_refined_parallax.png"] = artifact_description(
          true,
          false,
          "parallax-v2 full-resolution contour ownership refinement preview",
          "Finite p2-p98 grayscale preview of the ownership-refined candidate."
        );
        artifacts["shadow_ownership_refined_parallax_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 full-resolution contour ownership refinement preview",
          "Finite p2-p98 jet preview of the ownership-refined candidate."
        );
        artifacts["shadow_vertical_majorant.f32"] = hashed_artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate",
          video_region ?
            "Exact signed one-eye ROI-local source-U for the least column-wise upper envelope v+ >= ownership-refined candidate with |dv+/dy| <= max_vertical_shear/content_width; crop-local diagnostic evidence only." :
            "Exact signed one-eye full-source-U for the least column-wise upper envelope v+ >= ownership-refined candidate with |dv+/dy| <= max_vertical_shear/content_width; diagnostic evidence only.",
          shadow_vertical_majorant_sha256
        );
        artifacts["shadow_vertical_majorant_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate contract",
          video_region ?
            "Crop-local dimensions, ROI-local source-U units, finite scalar range, v >= candidate, and the generated max_vertical_shear bound; not full-source renderer authority." :
            "Full-source dimensions, full-source-U units, finite scalar range, v >= candidate, and the generated max_vertical_shear bound; not the live renderer position authority."
        );
        artifacts["shadow_vertical_majorant.png"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate preview",
          "Finite p2-p98 grayscale preview of the exact column-wise majorant."
        );
        artifacts["shadow_vertical_majorant_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 vertical shear-limiter intermediate preview",
          "Finite p2-p98 jet preview of the exact column-wise majorant."
        );
        artifacts["shadow_vertical_conditioned.f32"] = hashed_artifact_description(
          true,
          false,
          "parallax-v2 orientation-selective vertical conditioner",
          video_region ?
            "Exact signed one-eye ROI-local source-U after the fixed 75/25 share of the column upper/lower envelopes; may raise or lower the crop-local candidate while preserving the vertical shear bound." :
            "Exact signed one-eye full-source-U after the fixed 75/25 share of the column upper/lower envelopes; may raise or lower candidate while preserving the vertical shear bound.",
          shadow_vertical_conditioned_sha256
        );
        artifacts["shadow_vertical_conditioned_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 orientation-selective vertical conditioner contract",
          video_region ?
            "Crop-local dimensions, ROI-local source-U units, finite scalar range, authenticated envelope share, and vertical shear bound; intermediate consumed by the crop-local row majorant." :
            "Full-source dimensions, full-source-U units, finite scalar range, authenticated envelope share, and vertical shear bound; intermediate consumed by the row majorant."
        );
        artifacts["shadow_vertical_conditioned.png"] = artifact_description(
          true,
          false,
          "parallax-v2 orientation-selective vertical conditioner preview",
          "Finite p2-p98 grayscale preview of the exact vertical share."
        );
        artifacts["shadow_vertical_conditioned_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 orientation-selective vertical conditioner preview",
          "Finite p2-p98 jet preview of the exact vertical share."
        );
        if (subtitle_slr13_active) {
          artifacts["shadow_base_final_parallax.f32"] = hashed_artifact_description(
            true,
            true,
            "ordinary post-limiter V2 field before SLR13 conditioning",
            video_region ?
              "Exact ordinary crop-local row-majorant field before SLR13; not renderer authority without current SLR13 conditioning and depth_input_region embedding." :
              "Exact ordinary full-source-U row-majorant field before SLR13; not selected renderer authority.",
            shadow_base_final_sha256
          );
        }
        artifacts["shadow_final_parallax.f32"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 atomic final displacement field",
          video_region ?
            "Exact complete atomic one-eye ROI-local source-U field. Live renderer authority is this field together with depth_input_region embedding." :
            "Exact complete atomic one-eye full-source-U field; sole live V2 render position authority.",
          shadow_final_sha256
        );
        artifacts["shadow_final_parallax_shape.json"] = artifact_description(
          true,
          false,
          "parallax-v2 atomic final displacement contract",
          video_region ?
            "Crop-local dimensions, ROI-local source-U units, finite scalar range, authenticated vertical share, and limiter bounds; renderer authority only with depth_input_region.json." :
            "Dimensions, full-source-U units, finite scalar range, authenticated vertical share, horizontal slope bound, and vertical shear bound; live renderer authority."
        );
        artifacts["shadow_final_parallax.png"] = artifact_description(
          true,
          false,
          "parallax-v2 displayed displacement preview",
          video_region ?
            "Finite p2-p98 grayscale preview of crop-local q; full-source renderer authority additionally requires depth_input_region.json." :
            "Finite p2-p98 grayscale preview of the live V2 position field."
        );
        artifacts["shadow_final_parallax_heat.png"] = artifact_description(
          true,
          false,
          "parallax-v2 displayed displacement preview",
          video_region ?
            "Finite p2-p98 jet preview of crop-local q; full-source renderer authority additionally requires depth_input_region.json." :
            "Finite p2-p98 jet preview of the live V2 position field."
        );
        artifacts["shadow_state.json"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 shot calibration and attenuation state",
          "Independent typed state bound to the exact coordinate contract tag.",
          shadow_state_sha256
        );
        artifacts["shadow_frame_stats.json"] = hashed_artifact_description(
          true,
          true,
          "parallax-v2 current-frame moments",
          "Independent mean/std/min/max state bound to the exact coordinate contract tag.",
          shadow_frame_stats_sha256
        );
        artifacts["warp_map.f32"] = video_region ?
          hashed_artifact_description(
            true,
            true,
            "exact full-source inverse-warp mapping",
            "Raw normalized full-source U selected by production Reproject after crop-local field embedding and the outside-only zero-plane collar.",
            warp_map_sha256
          ) :
          artifact_description(
            warp_map_available,
            false,
            "exact inverse-warp mapping",
            warp_map_available ?
              "Raw normalized source-U selected by the production Reproject function." :
              "Unavailable because the matching dump-only mapping pass could not be created."
          );
        artifacts["warp_map_shape.json"] = artifact_description(
          warp_map_available,
          video_region,
          "inverse-warp mapping contract",
          warp_map_available ?
            "Dimensions, content fit, validity rules, and displacement derivation." :
            "Unavailable with the matching dump-only mapping pass."
        );
        artifacts["warp_displacement_heat.png"] = artifact_description(
          warp_map_available,
          false,
          "derived inverse displacement",
          warp_map_available ?
            "Signed output-eye-pixel displacement derived from the exact inverse map." :
            "Unavailable with the matching dump-only mapping pass."
        );
        artifacts["warp_mask.png"] = artifact_description(
          warp_mask_available,
          false,
          "V2 boundary-extrapolation mask",
          warp_mask_available ?
            "Red marks inverse samples outside the finite source interval that the live renderer clamps to the nearest boundary column; V2 has no internal owner selection or synthetic fill." :
            "Unavailable because the matching dump-only mask pass could not be created."
        );
      artifacts["sbs.png"] = artifact_description(
        true,
        true,
        "packed Host-SBS output",
        "Final packed stereo preview using the same matched-frame transfer handling as source.png."
      );
        artifacts["meta.txt"] = artifact_description(
          true,
          true,
          "human-readable summary",
          "Compact compatibility summary; dump_manifest.json is authoritative."
        );
        artifacts["dump_manifest.json"] = artifact_description(
          true,
          true,
          "package contract",
          "Authoritative dimensions, formats, settings, stage descriptions, and availability."
        );

        nlohmann::json dimensions {
          {"source", texture_description(source)},
          {"analysis_source", texture_description(depth_input_source)},
          {"model_input", {
                            {"width", completed.model_width},
                            {"height", completed.model_height},
                            {"channels", 3},
                            {"layout", "NCHW"},
                            {"dtype", "float32-le"},
                          }},
          {"raw_depth", {
                          {"width", completed.raw_width},
                          {"height", completed.raw_height},
                          {"format", "float32-le structured buffer"},
                        }},
          {"normalized_depth", nullptr},
          {"warp_depth", texture_description(warp_depth)},
          {"packed_sbs", texture_description(sbs)},
          {"eye", {
                    {"width", eye_width},
                    {"height", eye_height},
                  }},
          {"content_fit", {
                            {"scale_x", content_scale_x},
                            {"scale_y", content_scale_y},
                          }},
        };
        if (warp_map_available) {
          dimensions["warp_map"] = texture_description(warp_map);
        } else {
          dimensions["warp_map"] = nullptr;
        }
        if (warp_mask_available) {
          dimensions["warp_mask"] = texture_description(warp_mask);
        } else {
          dimensions["warp_mask"] = nullptr;
        }
        dimensions["shadow_coordinate"] = texture_description(shadow_coordinate);
        dimensions["shadow_candidate_parallax"] =
          texture_description(shadow_candidate);
        dimensions["shadow_ownership_refined_parallax"] =
          texture_description(shadow_ownership_refined);
        dimensions["shadow_vertical_majorant"] =
          texture_description(shadow_vertical);
        dimensions["shadow_vertical_conditioned"] =
          texture_description(shadow_vertical_conditioned);
        if (subtitle_slr13_active) {
          dimensions["shadow_base_final_parallax"] =
            texture_description(shadow_base_final);
        }
        dimensions["shadow_final_parallax"] = texture_description(shadow_final);

        const std::string color_mode =
          completed.color_space == models::input_color_space::scrgb_hdr  ? "scrgb_hdr" :
          completed.color_space == models::input_color_space::linear_sdr ? "linear_sdr" :
                                                                           "srgb";
        const std::string color_preview_transform =
          completed.color_space == models::input_color_space::scrgb_hdr ?
            "luminance-preserving diagnostic tone map then sRGB OETF" :
          completed.color_space == models::input_color_space::linear_sdr ?
            "sRGB OETF" :
            "none; values are already sRGB code values";
        const std::string trigger_source =
          by_button && by_file ? "button+file" : by_button ? "button" :
                                                             "file";
        const nlohmann::json shadow_shader_source =
          parallax_v2_shader_identity_json(
            *completed.parallax_v2_shader_provenance
          );
        nlohmann::json manifest {
          {"schema", 36},
          {"capture", "one matched, completed Host-SBS frame"},
          {"capture_status", "complete"},
          {"published_atomically", true},
          {"host_sbs_mode", "ai"},
          {"trigger", trigger_source},
          {"matched_frame_id", completed.matched_frame_id},
          {"depth_model", completed.depth_model},
          {"color_mode", color_mode},
          {"color_preview_transform", color_preview_transform},
          {"hdr_preview", hdr ? color_preview_transform : "not applied"},
          {"cuda_graph_active", completed.cuda_graph_active},
          {"warp_depth_prefilter_applied", false},
          {"renderer", {
                         {"authority", video_region ?
                           "authenticated crop-local atomic final field plus depth-input-region embedding" :
                           "authenticated-parallax-v2-atomic-final-field"},
                         {"parallax_v2_render_requested", true},
                         {"parallax_v2_render_selected", true},
                         {"mapping_artifacts_match_selected_renderer", warp_map_available && warp_mask_available},
                         {"parallax_v2_position_field", video_region ?
                           "shadow_final_parallax + depth_input_region embedding" :
                           "shadow_final_parallax"},
                         {"parallax_v2_coordinate_role", "shadow_coordinate is diagnostic only; it has no renderer authority"},
                         {"parallax_v2_ownership_refined_role", video_region ?
                           "conservative full-resolution crop-local source-contour foreground ownership applied to candidate before the vertical conditioner; may only raise uniquely owned far-side boundary texels" :
                           "conservative full-resolution source-contour foreground ownership applied to candidate before the vertical conditioner; may only raise uniquely owned far-side boundary texels"},
                         {"parallax_v2_vertical_majorant_role", "least column-wise upper envelope v+ >= ownership-refined candidate with adjacent-row source-U change <= max_vertical_shear/content_width; diagnostic evidence only"},
                         {"parallax_v2_vertical_conditioned_role", "fixed 75/25 share of column upper/lower envelopes; may raise or lower candidate and feeds the row majorant"},
                         {"parallax_v2_conditioner_role", subtitle_slr13_active ?
                           (video_region ?
                             "least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= max_horizontal_slope and vertical shear <= max_vertical_shear produces shadow_base_final_parallax; SLR13 publishes shadow_final_parallax atomically as direct live authority with depth_input_region embedding" :
                             "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= max_horizontal_slope and vertical shear <= max_vertical_shear produces shadow_base_final_parallax; SLR13 publishes shadow_final_parallax atomically as direct live authority") :
                           (video_region ?
                             "least row-wise crop-local q >= shadow_vertical_conditioned with horizontal slope <= max_horizontal_slope and vertical shear <= max_vertical_shear publishes shadow_final_parallax atomically as direct live authority with depth_input_region embedding" :
                             "least row-wise q >= shadow_vertical_conditioned with horizontal slope <= max_horizontal_slope and vertical shear <= max_vertical_shear publishes shadow_final_parallax atomically as direct live authority")},
                         {"parallax_v2_inverse", "11-step contractive fixed point; no forward-warp owner/visibility splat and no synthetic fill"},
                         {"collar_defocus", nlohmann::json {
                              {"enabled", false},
                              {"role", "disabled after live hand-boundary halo regression; live color uses one linear sample at the inverse-warped coordinate"},
                              {"kernel", "none"},
                              {"hdr", "native source sample; no clamp, tone map, or gamma conversion"},
                            }},
                         {"live_shader_source", nlohmann::json {
                              {"source_closure_schema", models::host_sbs_shader_cache::source_closure_schema},
                              {"source_compile_flags", models::host_sbs_shader_cache::shader_compile_flags},
                              {"source_macro_count", 0u},
                              {"source_closure_sha256", completed.parallax_v2_live_renderer_source_closure_sha256},
                              {"source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.filename}},
                              {"entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.entrypoint}},
                              {"target", std::string {models::host_sbs_shader_cache::parallax_v2_live_renderer.target}},
                              {"diagnostic_source_closure_sha256", std::string {models::host_sbs_shader_cache::parallax_v2_diagnostic_source_closure_sha256}},
                              {"mapping_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.filename}},
                              {"mapping_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mapping.entrypoint}},
                              {"mask_source_file", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.filename}},
                              {"mask_entrypoint", std::string {models::host_sbs_shader_cache::parallax_v2_live_mask.entrypoint}},
                            }},
                       }},
          {"dimensions", std::move(dimensions)},
          {"final_parallax", {
            {"contract_schema",
             models::depth_coordinate_v2::final_parallax_contract_schema},
            {"artifact", "shadow_final_parallax.f32"},
            {"warp_artifact", "warp_depth.f32"},
            {"authority", std::string {
              models::depth_coordinate_v2::final_parallax_authority
            }},
            {"publication_policy", std::string {
              models::depth_coordinate_v2::final_parallax_publication_policy
            }},
            {"reuse_policy", std::string {
              models::depth_coordinate_v2::final_parallax_reuse_policy
            }},
            {"invalid_policy", std::string {
              models::depth_coordinate_v2::final_parallax_invalid_policy
            }},
            {"current_rgb_policy", std::string {
              models::depth_coordinate_v2::final_parallax_current_rgb_policy
            }},
            {"warp_relation", "bit-identical"},
          }},
          {"normalization", adaptive_available ?
             nlohmann::json {
               {"role", "comparison-only scene-cut bridge evidence; no live V2 geometry authority"},
               {"effective_lower", normalization.lower},
               {"effective_upper", normalization.upper},
               {"initialized", normalization.initialized > 0.5f},
               {"initialized_value", normalization.initialized},
               {"frame_state", normalization_frame_state_name(normalization.frame_state)},
               {"frame_state_value", normalization.frame_state},
             } : nlohmann::json {nullptr}},
          {"raw_depth_statistics", {
                                     {"finite_count", raw_stats.finite_count},
                                     {"sample_count", static_cast<std::uint64_t>(raw_stats.width) * raw_stats.height},
                                     {"finite_fraction", static_cast<double>(raw_stats.finite_count) / (static_cast<double>(raw_stats.width) * raw_stats.height)},
                                     {"minimum", raw_stats.minimum},
                                     {"maximum", raw_stats.maximum},
                                     {"preview_low_p02", raw_stats.preview_low},
                                     {"preview_high_p98", raw_stats.preview_high},
                                   }},
          {"adaptive_summary", adaptive_available ? adaptive["decoded"] :
                                                     nlohmann::json {nullptr}},
          {"depth_input_region", {
            {"available", true},
            {"artifact", "depth_input_region.json"},
            {"mode", video_region ? "window-region" : "full-source"},
            {"geometry_authority", true},
            {"renderer_authority", true},
          }},
          {"window_region", {
            {"available", window_region_available},
            {"artifact", window_region_available ?
              nlohmann::json("window_region.json") : nlohmann::json(nullptr)},
            {"observer_status", completed.window_region_observer_status},
            {"mapping_status", completed.window_region_mapping_status},
            {"geometry_authority", false},
            {"renderer_authority", false},
          }},
          {"subtitle_conditioning", subtitle_conditioning},
          {"gpu_trace", gpu_trace.available ?
            gpu_trace.summary :
            nlohmann::json {
              {"available", false},
              {"required", false},
              {"rendering_authority", false},
              {"raw_artifact", nullptr},
              {"decoded_artifact", nullptr},
              {"contract_artifact", nullptr},
            }},
          {"parallax_v2_shadow", {
                                    {"requested", false},
                                    {"active", true},
                                    {"rendered_output_selected", true},
                                    {"shader_source", shadow_shader_source},
                                    {"state", std::move(shadow_summary)},
                                  }},
          {"config", config_json(
                       cfg,
                       completed,
                       completed.depth_model,
                       completed.raw_model_provenance->depth_model_url
                     )},
          {"artifacts", std::move(artifacts)},
        };
        manifest[std::string {
            models::depth_coordinate_v2::capture_provenance_manifest_key
          }] = {
            {"schema", models::depth_coordinate_v2::capture_provenance_schema},
            {"binding", std::string {
                          models::depth_coordinate_v2::capture_provenance_binding
                        }},
            {"depth_model", model_identity.depth_model},
            {"depth_model_url", model_identity.depth_model_url},
            {"onnx_sha256", model_identity.onnx_sha256},
            {"preprocess_profile", model_identity.preprocess_profile},
            {"preprocess_source_closure_sha256",
             model_identity.preprocess_source_closure_sha256},
            {"raw_depth_sha256", raw_depth_sha256},
            {"model_input_sha256", model_input_sha256},
            {"model_input_shape_sha256", model_input_shape_sha256},
          };

        std::ostringstream meta;
        meta.imbue(std::locale::classic());
        meta << std::setprecision(std::numeric_limits<float>::max_digits10)
             << "depth_model=" << completed.depth_model << '\n'
             << "color_mode=" << color_mode << '\n'
             << "trigger=" << trigger_source << '\n'
             << "matched_frame_id=" << completed.matched_frame_id << '\n'
             << "source_width=" << source.desc.Width << '\n'
             << "source_height=" << source.desc.Height << '\n'
             << "depth_input_mode="
             << (video_region ? "window-region" : "full-source") << '\n'
             << "depth_input_left=" << completed.depth_input_region.left << '\n'
             << "depth_input_top=" << completed.depth_input_region.top << '\n'
             << "depth_input_right=" << completed.depth_input_region.right << '\n'
             << "depth_input_bottom=" << completed.depth_input_region.bottom << '\n'
             << "depth_input_analysis_generation="
             << completed.depth_input_region.analysis_generation << '\n'
             << "depth_input_domain_reset="
             << (completed.input_domain_reset ? "true" : "false") << '\n'
             << "depth_input_region_sha256=" << depth_input_region_sha256 << '\n'
             << "packed_sbs_width=" << sbs.desc.Width << '\n'
             << "packed_sbs_height=" << sbs.desc.Height << '\n'
             << "eye_width=" << eye_width << '\n'
             << "eye_height=" << eye_height << '\n'
             << "model_input_width=" << completed.model_width << '\n'
             << "model_input_height=" << completed.model_height << '\n'
             << "raw_depth_width=" << raw_stats.width << '\n'
             << "raw_depth_height=" << raw_stats.height << '\n'
             << "raw_depth_finite_fraction="
             << static_cast<double>(raw_stats.finite_count) /
                  (static_cast<double>(raw_stats.width) * raw_stats.height)
             << '\n'
             << "raw_depth_min=" << raw_stats.minimum << '\n'
             << "raw_depth_max=" << raw_stats.maximum << '\n'
             << "raw_depth_preview_low_p02=" << raw_stats.preview_low << '\n'
             << "raw_depth_preview_high_p98=" << raw_stats.preview_high << '\n'
             << "cut_bridge_diagnostics_available="
             << (adaptive_available ? "true" : "false") << '\n'
             << "normalization_effective_lower=" << normalization.lower << '\n'
             << "normalization_effective_upper=" << normalization.upper << '\n'
             << "normalization_initialized=" << normalization.initialized << '\n'
             << "normalization_frame_state=" << normalization.frame_state << '\n'
             << "cuda_graph_active="
             << (completed.cuda_graph_active ? "true" : "false") << '\n'
             << "warp_depth_prefilter_applied=false\n"
             << "warp_map_available=" << (warp_map_available ? "true" : "false")
             << '\n'
             << "warp_mask_available=" << (warp_mask_available ? "true" : "false")
             << '\n'
             << "window_region_available="
             << (window_region_available ? "true" : "false") << '\n'
             << "window_region_observer_status="
             << completed.window_region_observer_status << '\n'
             << "window_region_mapping_status="
             << completed.window_region_mapping_status << '\n'
             << "parallax_v2_shadow_requested="
             << "false\n"
             << "parallax_v2_shadow_active=true\n"
             << "parallax_v2_render_requested=true\n"
             << "parallax_v2_render_selected=true\n"
             << "renderer_authority="
             << (video_region ?
                   "authenticated crop-local atomic final field plus depth-input-region embedding" :
                   "authenticated-parallax-v2-atomic-final-field")
             << '\n'
             << "final_parallax_warp_relation=bit-identical\n"
             << "collar_defocus_enabled=false\n"
             << "collar_defocus_kernel=none\n"
             << "parallax_v2_live_renderer_source_closure_sha256="
             << completed.parallax_v2_live_renderer_source_closure_sha256 << '\n'
             << "raw_model_provenance=authoritative\n"
             << "raw_model_preprocess_profile="
             << model_identity.preprocess_profile
             << '\n'
             << "raw_model_preprocess_source_closure_sha256="
             << model_identity.preprocess_source_closure_sha256
             << '\n';
        const auto &shader_identity = *completed.parallax_v2_shader_provenance;
        meta << "parallax_v2_shader_source_closure_schema="
             << shader_identity.source_closure_schema << '\n'
             << "parallax_v2_shader_source_compile_flags="
             << shader_identity.source_compile_flags << '\n'
             << "parallax_v2_shader_source_macro_count="
             << shader_identity.source_macro_count << '\n'
             << "parallax_v2_shader_source_closure_sha256="
             << shader_identity.source_closure_sha256 << '\n';
        if (
          !write_text(paths.temporary / "meta.txt", meta.str()) ||
          !write_json(paths.temporary / "dump_manifest.json", manifest)
        ) {
          break;
        }

        error.clear();
        std::filesystem::rename(paths.temporary, paths.final, error);
        if (error) {
          BOOST_LOG(warning) << "SBS debug dump: atomic directory publication failed: "
                             << error.message();
          break;
        }
        success = true;
      } while (false);

      if (!success) {
        error.clear();
        std::filesystem::remove_all(paths.temporary, error);
        BOOST_LOG(warning)
          << "SBS debug dump failed; request retained for a rate-limited retry."sv;
        result.error = "artifact publication failed";
        return result;
      }
    } catch (const std::exception &exception) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw; partial output removed and request retained: "
          << exception.what();
      } catch (...) {
      }
      result.error = exception.what();
      return result;
    } catch (...) {
      try {
        if (!paths.temporary.empty()) {
          std::error_code cleanup_error;
          std::filesystem::remove_all(paths.temporary, cleanup_error);
        }
      } catch (...) {
      }
      try {
        BOOST_LOG(warning)
          << "SBS debug dump transaction threw an unknown exception; partial output removed and request retained."sv;
      } catch (...) {
      }
      result.error = "unknown publication exception";
      return result;
    }

    if (by_file) {
      try {
        error.clear();
        std::filesystem::remove(trigger, error);
      } catch (...) {
        error = std::make_error_code(std::errc::not_enough_memory);
      }
      if (error) {
        result.trigger_remove_failed = true;
        try {
          BOOST_LOG(warning)
            << "SBS debug dump: could not remove dump.trigger; file polling disabled for this session: "
            << error.message();
        } catch (...) {
        }
      }
    }
    try {
      BOOST_LOG(info) << "SBS debug dump written to "sv << paths.final.string()
                      << " (model "sv << completed.depth_model << ", frame "
                      << completed.matched_frame_id << ')';
    } catch (...) {
    }
    result.success = true;
    result.published_path = paths.final;
    return result;
  }

  }  // namespace

  void dumper::poll_pending_readback(ID3D11DeviceContext *ctx) noexcept {
    if (!pending_gpu_capture_ || !ctx || !async_) {
      return;
    }

    const auto fail_capture = [this](const char *reason, const bool retryable = true) noexcept {
      if (!pending_gpu_capture_) {
        return;
      }
      const auto retry_token = pending_gpu_capture_->retry_token;
      const bool by_button = pending_gpu_capture_->job.by_button;
      const bool by_file = pending_gpu_capture_->job.by_file;
      const auto button = pending_gpu_capture_->job.button_request;
      pending_gpu_capture_.reset();
      prepared_frame_id_ = 0;
      retry_not_before_ = retryable ?
        std::chrono::steady_clock::now() + retry_backoff :
        std::chrono::steady_clock::time_point {};
      const bool rearmed = retryable && async_ && async_->record_publication_failure(
        retry_token, by_button, by_file, button
      );
      if (!retryable && async_) {
        // Invalidate only this capture's retry epoch. A later click may already have re-armed the
        // shared button latch after the original request was consumed and must remain pending.
        async_->cancel_retries(nullptr);
      }
      if (!retryable) {
        snapshot_armed_for_dump_ = false;
        if (by_file) {
          try {
            std::error_code error;
            std::filesystem::remove(dir_ / "dump.trigger", error);
            if (error) {
              file_trigger_enabled_ = false;
            }
          } catch (...) {
            file_trigger_enabled_ = false;
          }
        }
        file_trigger_pending_ = false;
      }
      try {
        BOOST_LOG(warning) << "SBS debug dump staged GPU readback failed; request "sv
                           << (rearmed ? "re-armed" :
                               (retryable ? "discarded after cancellation" :
                                            "discarded as deterministic invalid evidence"))
                           << ": " << reason;
      } catch (...) {
      }
    };

    try {
      auto &pending = *pending_gpu_capture_;
      if (!pending.gpu_ready) {
        if (!pending.completion) {
          fail_capture("completion query is unavailable");
          return;
        }
        const HRESULT query_status = ctx->GetData(
          pending.completion.Get(),
          nullptr,
          0,
          D3D11_ASYNC_GETDATA_DONOTFLUSH
        );
        if (query_status == S_FALSE) {
          return;
        }
        if (query_status != S_OK) {
          fail_capture("completion query failed");
          return;
        }
        const auto ready_observed = std::chrono::steady_clock::now();
        pending.gpu_ready = true;
        pending.gpu_ready_age_ms = std::chrono::duration<double, std::milli>(
          ready_observed - pending.submitted_at
        ).count();
        pending.collection_started_at = ready_observed;
        pending.completion.Reset();
      }

      const auto poll_started = std::chrono::steady_clock::now();
      ++pending.collection_poll_count;
      collection_budget budget(poll_started);
      auto &job = pending.job;
      const auto finish_poll = [&]() noexcept {
        pending.cpu_collection_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - poll_started
        ).count();
      };

      const auto collect_required = [&]
      (
        const collect_status_e status,
        const detail::collection_stage_e next
      ) noexcept {
        if (status == collect_status_e::ready) {
          pending.collection_stage = next;
          return true;
        }
        finish_poll();
        if (status == collect_status_e::failed) {
          fail_capture("required staging resource could not be mapped");
        }
        // A terminal EVENT should normally make every preceding staging resource mappable.
        // WAS_STILL_DRAWING and a deliberately partial chunk both retain the immutable batch.
        return false;
      };

      while (pending_gpu_capture_) {
        if (budget.exhausted()) {
          finish_poll();
          return;
        }

        using stage_e = detail::collection_stage_e;
        switch (pending.collection_stage) {
          case stage_e::shadow_state:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.shadow_state,
                    models::depth_coordinate_v2::state_float_count,
                    job.shadow_state,
                    budget
                  ),
                  stage_e::shadow_frame_stats
                )) return;
            break;

          case stage_e::shadow_frame_stats:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.shadow_frame_stats,
                    models::depth_coordinate_v2::frame_stats_float_count,
                    job.shadow_frame_stats,
                    budget
                  ),
                  stage_e::subtitle_ocr_record
                )) return;
            break;

          case stage_e::subtitle_ocr_record:
            if (!pending.subtitle_slr13_active) {
              pending.collection_stage = stage_e::gpu_trace_ring;
              break;
            }
            if (!collect_required(
                  collect_buffer(
                    ctx,
                    pending.subtitle_ocr_record,
                    job.subtitle_ocr_record,
                    budget
                  ),
                  stage_e::subtitle_locator_state
                )) return;
            break;

          case stage_e::subtitle_locator_state:
            if (!collect_required(
                  collect_buffer(
                    ctx,
                    pending.subtitle_locator_state,
                    job.subtitle_locator_state,
                    budget
                  ),
                  stage_e::gpu_trace_ring
                )) return;
            break;

          case stage_e::gpu_trace_ring:
            if (!pending.gpu_trace_requested) {
              pending.collection_stage = stage_e::validate_evidence;
              break;
            }
            {
              const auto status = collect_buffer(
                ctx,
                pending.gpu_trace_ring,
                job.gpu_trace_ring,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::validate_gpu_trace;
              } else if (status == collect_status_e::failed) {
                pending.gpu_trace_requested = false;
                pending.gpu_trace_ring = {};
                job.gpu_trace_ring.clear();
                pending.collection_stage = stage_e::validate_evidence;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::validate_gpu_trace:
            if (!detail::gpu_trace_ring_is_canonical(
                  job.gpu_trace_ring,
                  job.completed
                )) {
              BOOST_LOG(warning)
                << "SBS debug dump: optional GPU completion trace failed canonical "sv
                   "validation and was omitted; the core dump remains complete."sv;
              pending.gpu_trace_requested = false;
              job.gpu_trace_ring.clear();
            }
            pending.collection_stage = stage_e::validate_evidence;
            break;

          case stage_e::validate_evidence:
            if (!shadow_state_is_dumpable(
                  job.shadow_state,
                  job.completed.parallax_v2_raw_coordinate_scale
                )) {
              finish_poll();
              fail_capture(
                "selected completion is not a current valid V2 camera/output pair"
              );
              return;
            }
            if (pending.subtitle_slr13_active && !detail::subtitle_records_match_completion(
                  job.subtitle_ocr_record,
                  job.subtitle_locator_state,
                  job.gpu_trace_ring,
                  job.completed,
                  [&job]() {
                    models::depth_coordinate_v2::state_words_t state_words {};
                    std::memcpy(
                      state_words.data(), job.shadow_state.data(), sizeof(state_words)
                    );
                    return state_words[models::depth_coordinate_v2::confirmed_cut_count];
                  }()
                )) {
              finish_poll();
              fail_capture("OCR8/SLR13 record identity or layout is invalid", false);
              return;
            }
            pending.collection_stage = stage_e::scene_normalization;
            break;

          case stage_e::scene_normalization:
            if (!pending.scene_cut_bridge_requested) {
              pending.collection_stage = stage_e::model_input;
              break;
            }
            {
              const auto status = collect_buffer(
                ctx,
                pending.depth_frame_state,
                pending.normalization_state_bytes,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::scene_adaptive;
              } else if (status == collect_status_e::failed) {
                pending.scene_cut_bridge_requested = false;
                pending.depth_frame_state.resource.Reset();
                pending.adaptive_state.resource.Reset();
                pending.normalization_state_bytes.clear();
                job.adaptive_state.clear();
                pending.collection_stage = stage_e::model_input;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::scene_adaptive:
            {
              const auto status = collect_buffer(
                ctx,
                pending.adaptive_state,
                job.adaptive_state,
                budget
              );
              if (status == collect_status_e::ready) {
                pending.collection_stage = stage_e::scene_decode;
              } else if (status == collect_status_e::failed) {
                pending.scene_cut_bridge_requested = false;
                pending.adaptive_state.resource.Reset();
                pending.normalization_state_bytes.clear();
                job.adaptive_state.clear();
                pending.collection_stage = stage_e::model_input;
              } else {
                finish_poll();
                return;
              }
            }
            break;

          case stage_e::scene_decode:
            // The retained scene-cut bridge is optional comparison evidence. Malformed bytes do
            // not reject the exact-frame core package.
            job.scene_cut_bridge_state_available =
              decode_normalization_state(
                pending.normalization_state_bytes, job.normalization
              ) == depth_dumpability::valid;
            pending.normalization_state_bytes.clear();
            if (!job.scene_cut_bridge_state_available) {
              job.adaptive_state.clear();
            }
            pending.collection_stage = stage_e::model_input;
            break;

          case stage_e::model_input:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.model_input,
                    static_cast<std::size_t>(job.completed.model_width) *
                      static_cast<std::size_t>(job.completed.model_height) * 3u,
                    job.model_input,
                    budget
                  ),
                  stage_e::raw_depth
                )) return;
            break;

          case stage_e::raw_depth:
            if (!collect_required(
                  collect_float_buffer(
                    ctx,
                    pending.raw_depth,
                    static_cast<std::size_t>(job.completed.raw_width) *
                      static_cast<std::size_t>(job.completed.raw_height),
                    job.raw_depth,
                    budget
                  ),
                  stage_e::warp_depth
                )) return;
            break;

          case stage_e::warp_depth:
            if (!collect_required(
                  collect_texture(ctx, pending.warp_depth, job.warp_depth, budget),
                  stage_e::shadow_coordinate
                )) return;
            break;

          case stage_e::shadow_coordinate:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_coordinate, job.shadow_coordinate, budget
                  ),
                  stage_e::shadow_candidate
                )) return;
            break;

          case stage_e::shadow_candidate:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_candidate, job.shadow_candidate, budget
                  ),
                  stage_e::shadow_ownership_refined
                )) return;
            break;

          case stage_e::shadow_ownership_refined:
            if (!collect_required(
                  collect_texture(
                    ctx,
                    pending.shadow_ownership_refined,
                    job.shadow_ownership_refined,
                    budget
                  ),
                  stage_e::shadow_vertical
                )) return;
            break;

          case stage_e::shadow_vertical:
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_vertical, job.shadow_vertical, budget
                  ),
                  stage_e::shadow_vertical_conditioned
                )) return;
            break;

          case stage_e::shadow_vertical_conditioned:
            if (!collect_required(
                  collect_texture(
                    ctx,
                    pending.shadow_vertical_conditioned,
                    job.shadow_vertical_conditioned,
                    budget
                  ),
                  stage_e::shadow_base_final
                )) return;
            break;

          case stage_e::shadow_base_final:
            if (!pending.subtitle_slr13_active) {
              pending.collection_stage = stage_e::shadow_final;
              break;
            }
            if (!collect_required(
                  collect_texture(
                    ctx, pending.shadow_base_final, job.shadow_base_final, budget
                  ),
                  stage_e::shadow_final
                )) return;
            break;

          case stage_e::shadow_final:
            if (!collect_required(
                  collect_texture(ctx, pending.shadow_final, job.shadow_final, budget),
                  stage_e::source
                )) return;
            break;

          case stage_e::source:
            if (!collect_required(
                  collect_texture(ctx, pending.source, job.source, budget),
                  stage_e::depth_input_source
                )) return;
            break;

          case stage_e::depth_input_source:
            if (!pending.depth_input_source_available) {
              pending.collection_stage = stage_e::sbs;
              break;
            }
            if (!collect_required(
                  collect_texture(
                    ctx, pending.depth_input_source, job.depth_input_source, budget
                  ),
                  stage_e::sbs
                )) return;
            break;

          case stage_e::sbs:
            if (!collect_required(
                  collect_texture(ctx, pending.sbs, job.sbs, budget),
                  stage_e::warp_map
                )) return;
            break;

          case stage_e::warp_map:
            if (!job.warp_map_available) {
              pending.collection_stage = stage_e::warp_mask;
              break;
            }
            if (!collect_required(
                  collect_texture(ctx, pending.warp_map, job.warp_map, budget),
                  stage_e::warp_mask
                )) return;
            break;

          case stage_e::warp_mask:
            if (!job.warp_mask_available) {
              pending.collection_stage = stage_e::complete;
              break;
            }
            if (!collect_required(
                  collect_texture(ctx, pending.warp_mask, job.warp_mask, budget),
                  stage_e::complete
                )) return;
            break;

          case stage_e::complete:
            finish_poll();
            {
              const auto queued_frame_id = job.completed.matched_frame_id;
              const auto retry_token = pending.retry_token;
              const bool by_button = job.by_button;
              const bool by_file = job.by_file;
              const auto button = job.button_request;
              const double gpu_ready_age_ms = pending.gpu_ready_age_ms;
              const double collection_ms = pending.cpu_collection_ms;
              const auto collection_poll_count = pending.collection_poll_count;
              const double collection_wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - pending.collection_started_at
              ).count();
              auto cpu_job = std::move(job);
              pending_gpu_capture_.reset();

              const auto worker_state = async_;
              bool queued = false;
              try {
                std::function<void()> publish_task =
                  [job = std::move(cpu_job), worker_state, retry_token]() mutable {
                  dump_publish_result result;
                  try {
                    result = publish_captured_dump(job);
                  } catch (const std::exception &exception) {
                    result.error = exception.what();
                  } catch (...) {
                    result.error = "unknown background publication exception";
                  }
                  if (!result.success) {
                    const bool rearmed = worker_state->record_publication_failure(
                      retry_token,
                      job.by_button,
                      job.by_file,
                      job.button_request
                    );
                    try {
                      BOOST_LOG(warning)
                        << "SBS debug dump background publication failed; request "
                        << (rearmed ? "re-armed" : "discarded after cancellation")
                        << (result.error.empty() ? "." : ": " + result.error);
                    } catch (...) {
                    }
                  }
                  if (result.trigger_remove_failed) {
                    worker_state->record_trigger_remove_failure();
                  }
                };
                queued = worker_state->enqueue(std::move(publish_task));
              } catch (...) {
                // The exact-frame CPU snapshot is already complete, but allocating the queued
                // task itself can still fail. Preserve the request through the common retry path.
                queued = false;
              }
              if (!queued) {
                retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
                worker_state->record_publication_failure(
                  retry_token,
                  by_button,
                  by_file,
                  button
                );
                return;
              }

              try {
                BOOST_LOG(info)
                  << "SBS debug dump stable GPU snapshot queued for background publication (frame "sv
                  << queued_frame_id << ", GPU-ready age " << gpu_ready_age_ms
                  << " ms, CPU collection " << collection_ms << " ms across "
                  << collection_poll_count << " polls, ready-to-CPU age "
                  << collection_wall_ms << " ms)."sv;
              } catch (...) {
              }
            }
            return;
        }
      }
    } catch (const std::exception &exception) {
      try {
        BOOST_LOG(warning) << "SBS debug dump staging collection threw: "
                           << exception.what();
      } catch (...) {
      }
      fail_capture("CPU snapshot allocation or collection failed");
    } catch (...) {
      fail_capture("CPU snapshot collection failed with an unknown exception");
    }
  }

  bool dumper::maybe_dump(
    ID3D11Device *device,
    ID3D11DeviceContext *ctx,
    const frame &completed,
    const config::video_t::sbs_t &cfg
  ) {
    const bool by_file = file_trigger_pending_;
    if (!snapshot_armed_for_dump_ || !async_ || async_->busy() ||
        pending_gpu_capture_) {
      return false;
    }
    snapshot_armed_for_dump_ = false;
    detail::button_request_guard button_request(button_request_);
    if (!button_request.consumed() && !by_file) {
      return false;
    }
    try {
      const auto submission_started = std::chrono::steady_clock::now();
      const std::uint64_t retry_token = async_->allow_retries_and_token();
      if (
        !device || !ctx || !completed.source || !completed.depth_input_source ||
        !completed.model_input ||
        !completed.raw_depth || !completed.warp_depth || !completed.sbs ||
        completed.model_width <= 0 || completed.model_height <= 0 ||
        completed.raw_width <= 0 || completed.raw_height <= 0 ||
        !completed.raw_model_provenance ||
        completed.raw_model_provenance->depth_model.empty() ||
        completed.raw_model_provenance->depth_model != completed.depth_model ||
        completed.raw_model_provenance->onnx_sha256.empty() ||
        completed.raw_model_provenance->preprocess_source_closure_sha256.empty() ||
        completed.raw_width != completed.model_width ||
        completed.raw_height != completed.model_height ||
        prepared_frame_id_ != completed.matched_frame_id
      ) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      const auto input_region_error = depth_input_region_error(completed);
      if (!input_region_error.empty()) {
        BOOST_LOG(warning) << "SBS debug dump: authenticated input-region metadata is "sv
                              "incomplete or inconsistent; dump rejected: "sv
                           << input_region_error << '.';
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (!completed.parallax_v2_render_selected ||
          !completed.parallax_v2_producer_active ||
          !completed.shadow_candidate_parallax ||
          !completed.shadow_ownership_refined_parallax ||
          !completed.shadow_vertical_majorant || !completed.shadow_vertical_conditioned ||
          !completed.shadow_final_parallax || !completed.shadow_state ||
          !completed.shadow_frame_stats) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 renderer is not selected or has an incomplete "sv
             "authenticated resource set; dump rejected (legacy live rendering is unsupported)."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      const bool subtitle_ocr_present = completed.ocr_box_record != nullptr;
      const bool subtitle_locator_present = completed.subtitle_locator_state != nullptr;
      const bool subtitle_slr13_active =
        !completed.subtitle_work_suppressed &&
        subtitle_ocr_present && subtitle_locator_present;
      if (subtitle_ocr_present != subtitle_locator_present ||
          (subtitle_slr13_active && (
            !completed.shadow_base_final_parallax ||
            !models::depth_coordinate_v2::subtitle_ocr_field_is_calibrated(
              static_cast<std::uint32_t>(completed.model_width),
              static_cast<std::uint32_t>(completed.model_height)
            )
          ))) {
        BOOST_LOG(warning)
          << "SBS debug dump: OCR8/SLR13 resources are partial; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (!completed.shadow_coordinate) {
        BOOST_LOG(warning)
          << "SBS debug dump: the explicit Dump 3D canonical-coordinate snapshot is "sv
             "unavailable; live V2 rendering remains authenticated and unaffected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }
      if (completed.parallax_v2_live_renderer_source_closure_sha256 !=
            models::host_sbs_shader_cache::
              parallax_v2_live_renderer_source_closure_sha256) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 renderer source closure is missing or "sv
             "mismatched; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      const auto &model_identity = *completed.raw_model_provenance;
      const auto *capture_calibration =
        models::depth_coordinate_v2::find_capture_calibration(
          model_identity.depth_model,
          model_identity.depth_model_url,
          model_identity.onnx_sha256,
          model_identity.preprocess_profile,
          model_identity.preprocess_source_closure_sha256,
          static_cast<std::uint32_t>(completed.model_width),
          static_cast<std::uint32_t>(completed.model_height)
        );
      if (!capture_calibration) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 resources do not resolve to exactly one "sv
             "authenticated model/preprocess/shape calibration; dump rejected."sv;
        return false;
      }
      if (!parallax_v2_shader_identity_matches_contract(
            completed.parallax_v2_shader_provenance
          )) {
        BOOST_LOG(warning)
          << "SBS debug dump: production V2 resources have missing or mismatched "sv
             "shader-source provenance; dump rejected."sv;
        return false;
      }

      const std::uint64_t model_pixels =
        static_cast<std::uint64_t>(completed.model_width) *
        static_cast<std::uint64_t>(completed.model_height);
      const std::uint64_t raw_values =
        static_cast<std::uint64_t>(completed.raw_width) *
        static_cast<std::uint64_t>(completed.raw_height);
      if (model_pixels > SIZE_MAX / (3u * sizeof(float)) ||
          raw_values > SIZE_MAX / sizeof(float)) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      auto pending = std::make_unique<detail::pending_gpu_capture>();
      auto &job = pending->job;
      job.root = dir_;
      job.trigger = dir_ / "dump.trigger";
      job.button_request = button_request_;
      job.by_button = button_request.consumed();
      job.by_file = by_file;
      job.completed = completed;
      job.cfg = cfg;
      job.preprocess = &capture_calibration->preprocess;
      job.warp_map_available = completed.warp_map != nullptr;
      job.warp_mask_available = completed.warp_mask != nullptr;
      pending->subtitle_slr13_active = subtitle_slr13_active;
      pending->gpu_trace_requested = completed.gpu_trace_ring != nullptr &&
        gpu_trace_shader_identity_matches_contract(completed.gpu_trace_provenance);
      pending->depth_input_source_available =
        completed.depth_input_region.video_region;
      pending->retry_token = retry_token;
      if (completed.depth_input_region.video_region &&
          (!job.warp_map_available || !job.warp_mask_available)) {
        BOOST_LOG(warning)
          << "SBS debug dump: ROI completion lacks the required full-source inverse map "sv
             "or mask; dump rejected."sv;
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
      if (FAILED(device->CreateQuery(&query_desc, &pending->completion))) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        return false;
      }

      bool staged =
        stage_texture(device, ctx, completed.source, pending->source) &&
        stage_buffer(
          device,
          ctx,
          completed.model_input,
          static_cast<std::size_t>(model_pixels) * 3u * sizeof(float),
          pending->model_input
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.raw_depth,
          static_cast<std::size_t>(raw_values) * sizeof(float),
          pending->raw_depth
        ) &&
        stage_texture(device, ctx, completed.warp_depth, pending->warp_depth) &&
        stage_texture(device, ctx, completed.sbs, pending->sbs) &&
        stage_texture(
          device, ctx, completed.shadow_coordinate, pending->shadow_coordinate
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_candidate_parallax,
          pending->shadow_candidate
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_ownership_refined_parallax,
          pending->shadow_ownership_refined
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_vertical_majorant,
          pending->shadow_vertical
        ) &&
        stage_texture(
          device,
          ctx,
          completed.shadow_vertical_conditioned,
          pending->shadow_vertical_conditioned
        ) &&
        stage_texture(
          device, ctx, completed.shadow_final_parallax, pending->shadow_final
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.shadow_state,
          models::depth_coordinate_v2::state_float_count * sizeof(float),
          pending->shadow_state
        ) &&
        stage_buffer(
          device,
          ctx,
          completed.shadow_frame_stats,
          models::depth_coordinate_v2::frame_stats_float_count * sizeof(float),
          pending->shadow_frame_stats
        );
      if (staged && subtitle_slr13_active) {
        staged =
          stage_texture(
            device,
            ctx,
            completed.shadow_base_final_parallax,
            pending->shadow_base_final
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.ocr_box_record,
            subtitle_ocr_record_word_count * sizeof(std::uint32_t),
            pending->subtitle_ocr_record
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.subtitle_locator_state,
            subtitle_locator_state_word_count * sizeof(std::uint32_t),
            pending->subtitle_locator_state
          );
      }
      if (staged && pending->gpu_trace_requested) {
        pending->gpu_trace_requested = stage_buffer(
          device,
          ctx,
          completed.gpu_trace_ring,
          models::host_sbs_gpu_trace::ring_byte_count,
          pending->gpu_trace_ring
        );
        if (!pending->gpu_trace_requested) {
          pending->gpu_trace_ring = {};
        }
      }
      if (staged && pending->depth_input_source_available) {
        staged = stage_texture(
          device,
          ctx,
          completed.depth_input_source,
          pending->depth_input_source
        );
      }
      if (staged && job.warp_map_available) {
        staged = stage_texture(device, ctx, completed.warp_map, pending->warp_map);
      }
      if (staged && job.warp_mask_available) {
        staged = stage_texture(device, ctx, completed.warp_mask, pending->warp_mask);
      }
      if (!staged) {
        retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
        BOOST_LOG(warning)
          << "SBS debug dump: GPU staging submission failed; request retained for retry."sv;
        return false;
      }

      if (completed.adaptive_state && completed.depth_frame_state) {
        pending->scene_cut_bridge_requested =
          stage_buffer(
            device,
            ctx,
            completed.depth_frame_state,
            4u * sizeof(float),
            pending->depth_frame_state
          ) &&
          stage_buffer(
            device,
            ctx,
            completed.adaptive_state,
            sbs_adaptive_state::word_count * sizeof(std::uint32_t),
            pending->adaptive_state
          );
        if (!pending->scene_cut_bridge_requested) {
          pending->depth_frame_state = {};
          pending->adaptive_state = {};
        }
      }

      // No later frame can race these bytes: every CopyResource precedes this event, and all
      // future live writes are submitted after it on the same immediate-context command stream.
      ctx->End(pending->completion.Get());
      pending->submitted_at = std::chrono::steady_clock::now();

      // No background code may observe a live COM pointer. The pending GPU object owns only
      // staging resources; the eventual publication job owns only copied scalar metadata.
      job.completed.source = nullptr;
      job.completed.depth_input_source = nullptr;
      job.completed.model_input = nullptr;
      job.completed.raw_depth = nullptr;
      job.completed.warp_depth = nullptr;
      job.completed.adaptive_state = nullptr;
      job.completed.depth_frame_state = nullptr;
      job.completed.warp_map = nullptr;
      job.completed.warp_mask = nullptr;
      job.completed.sbs = nullptr;
      job.completed.shadow_coordinate = nullptr;
      job.completed.shadow_candidate_parallax = nullptr;
      job.completed.shadow_ownership_refined_parallax = nullptr;
      job.completed.shadow_vertical_majorant = nullptr;
      job.completed.shadow_vertical_conditioned = nullptr;
      job.completed.shadow_base_final_parallax = nullptr;
      job.completed.shadow_final_parallax = nullptr;
      job.completed.ocr_box_record = nullptr;
      job.completed.subtitle_locator_state = nullptr;
      job.completed.gpu_trace_ring = nullptr;
      job.completed.shadow_state = nullptr;
      job.completed.shadow_frame_stats = nullptr;

      const auto submitted_frame_id = job.completed.matched_frame_id;
      pending_gpu_capture_ = std::move(pending);
      button_request.commit();
      if (by_file) {
        file_trigger_pending_ = false;
      }
      prepared_frame_id_ = 0;
      try {
        const double submission_ms = std::chrono::duration<double, std::milli>(
          pending_gpu_capture_->submitted_at - submission_started
        ).count();
        BOOST_LOG(info)
          << "SBS debug dump exact-frame GPU staging submitted without waiting (frame "sv
          << submitted_frame_id << ", CPU allocation/submission " << submission_ms
          << " ms)."sv;
      } catch (...) {
      }
      return true;
    } catch (const std::exception &exception) {
      retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump render-thread staging submission failed; request retained: "
          << exception.what();
      } catch (...) {
      }
      return false;
    } catch (...) {
      retry_not_before_ = std::chrono::steady_clock::now() + retry_backoff;
      try {
        BOOST_LOG(warning)
          << "SBS debug dump render-thread staging submission failed with an unknown "sv
             "exception; request retained."sv;
      } catch (...) {
      }
      return false;
    }
  }

}  // namespace platf::sbs_debug
