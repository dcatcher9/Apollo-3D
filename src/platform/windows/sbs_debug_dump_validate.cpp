/**
 * @file src/platform/windows/sbs_debug_dump_validate.cpp
 * @brief Dump 3D record validators: pure checks of captured bytes, without D3D work or file I/O.
 */
#include "sbs_debug_dump_validate.h"

// standard includes
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

// local includes
#include "src/depth_coordinate_v2.h"
#include "src/host_sbs_contour_history_readback.h"
#include "src/host_sbs_gpu_trace.h"
#include "src/host_sbs_subtitle_glyph.h"
#include "src/host_sbs_subtitle_plane.h"
#include "src/model_manager.h"
#include "src/prod_zipdepth_convex2x.h"
#include "sbs_debug_dump.h"

namespace platf::sbs_debug {

  namespace {

    using namespace models::depth_coordinate_v2;
    using detail::subtitle_u64;
    using detail::subtitle_word;
    using models::contour_history_readback::half_to_float;

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

    bool subtitle_ocr_classifier_history_is_canonical(
      const std::vector<std::uint8_t> &ocr,
      const frame &completed,
      const models::depth_tensor_content_rect_t tensor_content
    ) {
      const auto history = [&ocr](const std::size_t index) {
        return subtitle_word(ocr, subtitle_ocr_classifier_history_offset + index);
      };
      const auto history_u64 = [&ocr](const std::size_t index) {
        return subtitle_u64(ocr, subtitle_ocr_classifier_history_offset + index);
      };
      bool empty = true;
      for (std::size_t index = 0u; index < subtitle_ocr_classifier_history_word_count; ++index) {
        empty = empty && history(index) == 0u;
      }
      // A cold start or an invalid source clock may classify current geometry without retaining
      // a layout prior. Abstention never publishes history from a previous observation.
      if (empty) return true;
      if (subtitle_word(ocr, 2u) != 1u) return false;

      const auto source_width = subtitle_word(ocr, 9u);
      const auto source_height = subtitle_word(ocr, 10u);
      const auto crop_height = models::subtitle_ocr_source_crop_height(source_width, source_height);
      const auto reference = models::subtitle_ocr_line_reference(
        source_width, source_height,
        completed.depth_input_region.source_width,
        completed.depth_input_region.source_height,
        completed.depth_input_region.left, completed.depth_input_region.top,
        completed.depth_input_region.is_video_region()
      );
      if (!reference || crop_height == 0u ||
          history(0u) != subtitle_ocr_classifier_history_tag ||
          history(1u) != subtitle_ocr_classifier_history_schema ||
          history(2u) != subtitle_ocr_classifier_history_known_flags ||
          history(3u) > subtitle_ocr_classifier_prior_cap ||
          history_u64(4u) == 0u ||
          history_u64(4u) != completed.matched_observation_timestamp_us ||
          history_u64(6u) != subtitle_u64(ocr, 5u) ||
          history_u64(8u) != subtitle_u64(ocr, 7u) ||
          history(10u) != source_width || history(11u) != source_height ||
          history(12u) != source_height - crop_height || history(13u) != crop_height ||
          history(14u) != subtitle_word(ocr, 11u) || history(15u) != subtitle_word(ocr, 12u) ||
          history(16u) != tensor_content.left || history(17u) != tensor_content.top ||
          history(18u) != tensor_content.right || history(19u) != tensor_content.bottom ||
          history(20u) != std::bit_cast<std::uint32_t>(reference->center_twice_source) ||
          history(21u) != reference->width ||
          history(22u) != std::bit_cast<std::uint32_t>(reference->search_top_source) ||
          history(23u) > 1u || history(24u) > 1u ||
          (history(24u) == 0u ? history_u64(25u) != 0u :
            history_u64(25u) == 0u || history_u64(25u) > history_u64(4u)) ||
          history(30u) != 0u || history(31u) != 0u) {
        return false;
      }
      const float trusted_height = std::bit_cast<float>(history(27u));
      if (!std::isfinite(trusted_height) || trusted_height < 0.0f ||
          trusted_height > static_cast<float>(subtitle_ocr_safe_row_bottom - subtitle_ocr_safe_row_top) ||
          (trusted_height == 0.0f ? history(28u) != 0u || history(29u) != 0u :
            history(28u) < subtitle_ocr_safe_row_top || history(28u) >= history(29u) ||
              history(29u) > subtitle_ocr_safe_row_bottom)) {
        return false;
      }
      bool have_votes = false;
      for (std::size_t bin = 0u; bin < subtitle_ocr_classifier_prior_bin_count; ++bin) {
        const auto vote = history(32u + bin);
        if (vote > subtitle_ocr_classifier_prior_cap) return false;
        have_votes = have_votes || vote != 0u;
      }
      return !have_votes || (trusted_height > 0.0f && history(3u) != 0u);
    }

    models::host_sbs_subtitle_plane::words_t subtitle_plane_words(
      const std::vector<std::uint8_t> &bytes
    ) noexcept {
      models::host_sbs_subtitle_plane::words_t words {};
      if (bytes.size() == sizeof(words)) std::memcpy(words.data(), bytes.data(), sizeof(words));
      return words;
    }

    bool subtitle_ui_empty_has_no_hidden_plane(
      const models::host_sbs_subtitle_plane::words_t &words
    ) noexcept {
      const auto a = subtitle_plane_adaptive_offset;
      return words[a + 2u] == 0u && words[a + 3u] == 0u &&
        (words[a + 1u] & 4u) == 0u && words[a + 13u] == 0u &&
        words[a + 14u] == 0u && words[a + 15u] == 0u;
    }

  }  // namespace

  namespace detail {

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

    bool subtitle_ocr_record_is_canonical_for_frame(
      const std::vector<std::uint8_t> &ocr,
      const frame &completed
    ) {
      if (ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t)) {
        return false;
      }
      const auto field_width = static_cast<std::uint32_t>(completed.field_width);
      const auto field_height = static_cast<std::uint32_t>(completed.field_height);
      const auto geometry = models::fit_subtitle_analysis_geometry(
        completed.depth_input_region.width(),
        completed.depth_input_region.height(),
        {completed.field_width, completed.field_height},
        completed.field_content
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
          ) || !subtitle_ocr_pairs_are_canonical(ocr, final_count) ||
          !subtitle_ocr_classifier_history_is_canonical(ocr, completed, tensor_content)) {
        return false;
      }

      return true;
    }

    bool subtitle_records_match_frame(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &plane,
      const frame &completed
    ) {
      namespace ui = models::host_sbs_subtitle_plane;
      if (!joint_plane_mode_word_is_valid(completed.parallax_v2_joint_plane_mode) ||
          !std::isfinite(completed.parallax_v2_requested_gain) || completed.parallax_v2_requested_gain <= 0.0f ||
          completed.matched_frame_id == 0u || completed.matched_observation_timestamp_us == 0u ||
          !subtitle_ocr_record_is_canonical_for_frame(ocr, completed) || plane.size() != sizeof(ui::words_t)) return false;
      const auto words = subtitle_plane_words(plane);
      const auto geometry = models::fit_subtitle_analysis_geometry(
        completed.depth_input_region.width(), completed.depth_input_region.height(),
        {completed.field_width, completed.field_height}, completed.field_content);
      if (!geometry.valid() || !ui::state_words_are_canonical(words, display_budget()) ||
          ui::join(words.data() + subtitle_plane_frame_offset) != completed.matched_frame_id ||
          ui::join(words.data() + subtitle_plane_analysis_generation_offset) != completed.depth_input_region.analysis_generation ||
          ui::join(words.data() + subtitle_plane_observation_timestamp_offset) != completed.matched_observation_timestamp_us ||
          words[subtitle_plane_source_extent_offset] != completed.depth_input_region.width() ||
          words[subtitle_plane_source_extent_offset + 1u] != completed.depth_input_region.height() ||
          words[subtitle_plane_field_extent_offset] != static_cast<std::uint32_t>(completed.field_width) ||
          words[subtitle_plane_field_extent_offset + 1u] != static_cast<std::uint32_t>(completed.field_height) ||
          words[subtitle_plane_roi_offset] != geometry.roi_top || words[subtitle_plane_roi_offset + 1u] != geometry.roi_bottom ||
          words[subtitle_plane_content_offset] != completed.field_content.left ||
          words[subtitle_plane_content_offset + 1u] != completed.field_content.top ||
          words[subtitle_plane_content_offset + 2u] != completed.field_content.right ||
          words[subtitle_plane_content_offset + 3u] != completed.field_content.bottom) return false;
      const bool current_valid = words[subtitle_plane_flags_word] == subtitle_plane_current_valid_flag;
      const auto count = subtitle_word(ocr, 4u);
      // UIP4 contains no rectangle copy or component choice. Each admitted slot is the exact
      // independently authenticated current OCR8 pair in producer order, including legal empty.
      return current_valid ? subtitle_word(ocr, 2u) == 1u &&
        words[subtitle_plane_current_slot_mask_word] == (1u << count) - 1u :
        words[subtitle_plane_current_slot_mask_word] == 0u;
    }

    bool gpu_trace_ring_is_canonical(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    ) {
      using namespace models::host_sbs_gpu_trace;
      if (ring.size() != ring_byte_count || completed.matched_frame_id == 0u ||
          completed.model_width <= 0 || completed.model_height <= 0 ||
          completed.field_width <= 0 || completed.field_height <= 0 ||
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
      std::optional<models::host_sbs_subtitle_plane::words_t> previous_ui_plane;
      std::uint64_t previous_ui_domain_tag = 0u;
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
          for (std::size_t index = 0u; index < models::depth_coordinate_v2::subtitle_condition_param_word_count; ++index) {
            if (word(base + word_index(record_word_e::subtitle_condition_begin) + index) !=
                word(previous_base +
                     word_index(record_word_e::subtitle_condition_begin) + index)) {
              return false;
            }
          }
        }

        const bool suppressed = (flags & subtitle_suppressed) != 0u;
        if (!suppressed) {
          namespace ui = models::host_sbs_subtitle_plane;
          const auto plane_base = base + word_index(record_word_e::subtitle_locator_begin);
          const auto condition_base = base + word_index(record_word_e::subtitle_condition_begin);
          ui::words_t plane {};
          ui::condition_words_t condition {};
          for (std::size_t index = 0u; index < plane.size(); ++index) plane[index] = word(plane_base + index);
          for (std::size_t index = 0u; index < condition.size(); ++index) condition[index] = word(condition_base + index);
          const auto published_frame = ui::join(plane.data() + subtitle_plane_frame_offset);
          const auto published_time = ui::join(plane.data() + subtitle_plane_observation_timestamp_offset);
          const auto content = models::depth_tensor_content_rect_t {
            plane[subtitle_plane_content_offset], plane[subtitle_plane_content_offset + 1u],
            plane[subtitle_plane_content_offset + 2u], plane[subtitle_plane_content_offset + 3u],
          };
          const auto geometry = models::fit_subtitle_analysis_geometry(
            plane[subtitle_plane_source_extent_offset], plane[subtitle_plane_source_extent_offset + 1u],
            {static_cast<int>(plane[subtitle_plane_field_extent_offset]), static_cast<int>(plane[subtitle_plane_field_extent_offset + 1u])}, content);
          if (!ui::state_words_are_canonical(plane, display_budget()) ||
              !geometry.valid() ||
              (subtitle_disposition == subtitle_disposition_e::abstention &&
               (plane[subtitle_plane_flags_word] != 0u || plane[subtitle_plane_current_slot_mask_word] != 0u)) ||
              ui::join(plane.data() + subtitle_plane_analysis_generation_offset) != analysis_generation ||
              plane[subtitle_plane_source_extent_offset] != record(record_word_e::source_width) ||
              plane[subtitle_plane_source_extent_offset + 1u] != record(record_word_e::source_height) ||
              plane[subtitle_plane_field_extent_offset] != record(record_word_e::field_width) ||
              plane[subtitle_plane_field_extent_offset + 1u] != record(record_word_e::field_height) ||
              plane[subtitle_plane_roi_offset] != geometry.roi_top || plane[subtitle_plane_roi_offset + 1u] != geometry.roi_bottom ||
              (subtitle_held_with_depth ? published_frame == 0u || published_frame >= frame_id ||
                 published_time == 0u || published_time > observation_timestamp_us :
                 published_frame != frame_id || published_time != observation_timestamp_us) ||
              !ui::condition_words_match_plane(condition, plane, true)) return false;
          // A standalone snapshot cannot distinguish a genuine empty from a clock-fault
          // redispatch with held hidden state. Only a proven new current observation (or an
          // explicit domain reset) must have cleared that invisible plane immediately.
          const bool explicit_reset = (flags & input_domain_reset) != 0u ||
            (previous_ui_plane && previous_ui_domain_tag != domain_tag);
          const bool genuinely_newer = previous_ui_plane &&
            previous_ui_domain_tag == domain_tag &&
            published_frame != ui::join(previous_ui_plane->data() + subtitle_plane_frame_offset) &&
            published_time > ui::join(previous_ui_plane->data() + subtitle_plane_adaptive_offset + 4u);
          if (!subtitle_held_with_depth &&
              plane[subtitle_plane_flags_word] == subtitle_plane_current_valid_flag &&
              ui::current_count(plane) == 0u &&
              ui::join(plane.data() + subtitle_plane_adaptive_offset + 4u) == published_time &&
              (explicit_reset || genuinely_newer) && !subtitle_ui_empty_has_no_hidden_plane(plane)) return false;
          previous_ui_plane = plane;
          previous_ui_domain_tag = domain_tag;
          // Historical UIP4 snapshots prove identity/controller state and SCP4, not OCR cover
          // geometry. Exact paired covers are joined only from the separately captured OCR8.
        } else {
          previous_ui_plane.reset();
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
                static_cast<std::uint32_t>(completed.field_width) ||
              record(record_word_e::field_height) !=
                static_cast<std::uint32_t>(completed.field_height) ||
              (completed.matched_observation_timestamp_us != 0u &&
               observation_timestamp_us != completed.matched_observation_timestamp_us) ||
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
      const frame &completed
    ) {
      using namespace models::host_sbs_gpu_trace;
      if (subtitle_records_match_frame(ocr, locator, completed)) {
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
          locator.size() != subtitle_plane_state_word_count * sizeof(std::uint32_t)) {
        return false;
      }
      const auto held_frame_id = subtitle_u64(locator, subtitle_plane_frame_offset);
      if (held_frame_id == 0u || held_frame_id >= completed.matched_frame_id) {
        return false;
      }
      frame held = completed;
      held.matched_frame_id = held_frame_id;
      held.matched_observation_timestamp_us = subtitle_u64(locator, subtitle_plane_observation_timestamp_offset);
      if (held.matched_observation_timestamp_us == 0u ||
          held.matched_observation_timestamp_us > completed.matched_observation_timestamp_us) return false;
      held.input_domain_reset = false;
      held.gpu_undecided_completion = false;
      if (!subtitle_records_match_frame(ocr, locator, held)) {
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
      for (std::size_t index = 0u; index < subtitle_plane_state_word_count;
           ++index) {
        if (word(*matched_locator_base + index) != subtitle_word(locator, index)) {
          return false;
        }
      }
      return true;
    }

    bool parallax_v2_state_matches_frame(
      const std::vector<float> &state,
      const std::vector<float> &frame_stats,
      const frame &completed
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      if (state.size() != state_float_count || frame_stats.size() != frame_stats_float_count) {
        return false;
      }
      for (std::size_t index = 0; index < state.size(); ++index) {
        if (state_fields[index].gpu_encoding == state_gpu_encoding_e::float_value &&
            !std::isfinite(state[index])) return false;
      }
      if (!std::all_of(frame_stats.begin(), frame_stats.end(), [](float value) {
            return std::isfinite(value);
          })) return false;
      state_words_t words {};
      std::memcpy(words.data(), state.data(), sizeof(words));
      if (!parallax_runtime_constants_are_valid(
            completed.parallax_v2_raw_coordinate_scale,
            completed.parallax_v2_requested_pop_strength,
            completed.parallax_v2_requested_gain) ||
          !parallax_state_words_are_authenticated(
            words, completed.parallax_v2_raw_coordinate_scale,
            completed.parallax_v2_joint_plane_mode)) return false;
      const bool state_valid = state[frame_valid] == 1.0f;
      const bool statistics_valid = frame_stats[frame_stat_valid] == 1.0f;
      const float valid_count = frame_stats[frame_stat_valid_count];
      const float texel_count = frame_stats[frame_stat_texel_count];
      const bool counts_valid = valid_count >= 0.0f && texel_count > 0.0f &&
        valid_count <= texel_count && valid_count == std::floor(valid_count) &&
        texel_count == std::floor(texel_count) &&
        texel_count <= static_cast<float>(std::numeric_limits<std::uint32_t>::max());
      const bool statistics_canonical =
        (frame_stats[frame_stat_valid] == 0.0f || frame_stats[frame_stat_valid] == 1.0f) &&
        counts_valid && frame_stats[frame_stat_population_std] >= 0.0f &&
        (statistics_valid ?
          (valid_count == texel_count && frame_stats[frame_stat_maximum] >= frame_stats[frame_stat_minimum]) :
          (frame_stats[frame_stat_mean] == 0.0f && frame_stats[frame_stat_population_std] == 0.0f &&
           frame_stats[frame_stat_minimum] == 0.0f && frame_stats[frame_stat_maximum] == 0.0f));
      if (!statistics_canonical) return false;
      const float percentile_valid = frame_stats[frame_stat_percentile_valid];
      const float percentile_low = frame_stats[frame_stat_percentile_low];
      const float percentile_high = frame_stats[frame_stat_percentile_high];
      const float percentile_bin_width = frame_stats[frame_stat_percentile_bin_width];
      if ((percentile_valid != 0.0f && percentile_valid != 1.0f) ||
          (percentile_valid == 1.0f ?
            (!statistics_valid || percentile_bin_width < 0.0f ||
             percentile_low < frame_stats[frame_stat_minimum] ||
             percentile_high > frame_stats[frame_stat_maximum] || percentile_low > percentile_high) :
            (percentile_low != 0.0f || percentile_high != 0.0f || percentile_bin_width != 0.0f))) {
        return false;
      }
      const bool usable = statistics_valid && frame_stats[frame_stat_population_std] > collapse_abs_epsilon;
      const float budget = display_budget();
      if ((state[gain_display_limit] != 0.0f || state_valid) && state[gain_display_limit] != budget) {
        return false;
      }
      // Missing or regressed source time can disarm an established camera despite usable
      // current statistics. The sealed state admission still forbids false valid output.
      const bool quantiles_valid = percentile_valid == 1.0f;
      return !state_valid || (usable && quantiles_valid);
    }

    bool composite_runtime_provenance_is_authenticated(const frame &completed) noexcept {
      try {
        if (!completed.refined_live_geometry_active) {
          return !completed.composite_depth_runtime_provenance;
        }
        if (!completed.raw_model_provenance ||
            !completed.composite_depth_runtime_provenance) {
          return false;
        }
        const auto &raw = *completed.raw_model_provenance;
        const auto &composite = *completed.composite_depth_runtime_provenance;
        const std::string active_manifest =
          std::string {models::prod_zipdepth_convex2x::logical_model} +
          ".active-engine.json";
        return composite.model == models::prod_zipdepth_convex2x::logical_model &&
               composite.onnx_sha256 ==
                 models::prod_zipdepth_convex2x::fused_onnx_sha256 &&
               composite.embedded_dav2_onnx_sha256 ==
                 models::prod_zipdepth_convex2x::dav2_onnx_sha256 &&
               composite.embedded_dav2_onnx_sha256 == raw.onnx_sha256 &&
               composite.zipdepth_checkpoint_sha256 ==
                 models::prod_zipdepth_convex2x::zipdepth_checkpoint_sha256 &&
               composite.guidance_preprocess_source_closure_sha256 ==
                 raw.preprocess_source_closure_sha256 &&
               composite.engine_recipe ==
                 models::prod_zipdepth_convex2x::engine_recipe &&
               models::is_current_depth_engine_filename(composite.engine_artifact) &&
               composite.active_engine_manifest == active_manifest;
      } catch (...) {
        return false;
      }
    }

    bool subtitle_contour_texture_layout_is_authenticated(
      const subtitle_contour_texture_e role,
      const DXGI_FORMAT format,
      const std::uint32_t width,
      const std::uint32_t height,
      const std::size_t row_bytes,
      const std::size_t byte_count,
      const frame &completed
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      const auto source_width = completed.depth_input_region.width();
      const auto source_height = completed.depth_input_region.height();
      if (source_width == 0u || source_height == 0u || completed.field_width <= 0 ||
          completed.field_height <= 0) return false;
      const bool supported = source_width <= subtitle_contour_source_max_extent &&
                             source_height <= subtitle_contour_source_max_extent;
      const bool field = role == subtitle_contour_texture_e::field_pin ||
                         role == subtitle_contour_texture_e::distance ||
                         role == subtitle_contour_texture_e::gaussian_weight ||
                         role == subtitle_contour_texture_e::gaussian_blend ||
                         role == subtitle_contour_texture_e::gaussian_vertical_conditioned;
      const auto expected_width = field ? static_cast<std::uint32_t>(completed.field_width) :
                                         (supported ? source_width : 1u);
      const auto expected_height = field ? static_cast<std::uint32_t>(completed.field_height) :
        (supported ? models::subtitle_ocr_source_crop_height(source_width, source_height) : 1u);
      DXGI_FORMAT expected_format = DXGI_FORMAT_R32_UINT;
      std::size_t pixel_bytes = sizeof(std::uint32_t);
      if (field && role != subtitle_contour_texture_e::field_pin) {
        expected_format = DXGI_FORMAT_R32_FLOAT;
      } else if (role == subtitle_contour_texture_e::history) {
        expected_format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pixel_bytes = 4u * sizeof(float);
      } else if (role == subtitle_contour_texture_e::source) {
        expected_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        pixel_bytes = 4u * sizeof(std::uint16_t);
      } else if (!field && role != subtitle_contour_texture_e::support && role != subtitle_contour_texture_e::glyph_matte) {
        return false;
      }
      return width == expected_width && height == expected_height && format == expected_format &&
             row_bytes == static_cast<std::size_t>(expected_width) * pixel_bytes &&
             byte_count == row_bytes * expected_height;
    }

    bool subtitle_contour_identity_matches_completion(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const std::vector<float> &state,
      const frame &completed
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      if (identity.size() != subtitle_contour_history_word_count * sizeof(std::uint32_t) ||
          ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t) ||
          locator.size() != subtitle_plane_state_word_count * sizeof(std::uint32_t) ||
          state.size() != state_float_count) return false;
      const auto word = [&identity](std::size_t index) { return subtitle_word(identity, index); };
      state_words_t geometry {};
      std::memcpy(geometry.data(), state.data(), sizeof(geometry));
      const auto owner_frame = subtitle_u64(ocr, 5u);
      const auto owner_clock = static_cast<std::uint64_t>(geometry[gain_last_observation_low]) |
        (static_cast<std::uint64_t>(geometry[gain_last_observation_high]) << 32u);
      const auto source_width = completed.depth_input_region.width();
      const auto source_height = completed.depth_input_region.height();
      const auto crop_height = models::subtitle_ocr_source_crop_height(source_width, source_height);
      const auto analysis = models::fit_subtitle_analysis_geometry(
        source_width, source_height, {completed.field_width, completed.field_height}, completed.field_content);
      if (!analysis.valid() || source_width == 0u || source_height == 0u ||
          word(0u) != subtitle_contour_schema || word(1u) != subtitle_contour_history_tag ||
          word(2u) > 1u || word(subtitle_contour_history_snapshot_valid_word) > 1u ||
          word(22u) != subtitle_contour_recipe ||
          subtitle_u64(identity, 4u) != owner_frame || subtitle_u64(locator, subtitle_plane_frame_offset) != owner_frame ||
          subtitle_u64(identity, 6u) != completed.depth_input_region.analysis_generation ||
          subtitle_u64(identity, 6u) != subtitle_u64(ocr, 7u) ||
          word(8u) != source_width || word(9u) != source_height ||
          word(10u) != source_height - crop_height || word(11u) != crop_height ||
          word(12u) != static_cast<std::uint32_t>(completed.field_width) ||
          word(13u) != static_cast<std::uint32_t>(completed.field_height) ||
          word(14u) != analysis.roi_top || word(15u) != analysis.roi_bottom ||
          word(16u) != completed.field_content.left || word(17u) != completed.field_content.top ||
          word(18u) != completed.field_content.right || word(19u) != completed.field_content.bottom) return false;
      for (std::size_t index = 8u; index < 10u; ++index) {
        if (word(index) != subtitle_word(ocr, index + 1u)) return false;
      }
      for (std::size_t index = 12u; index < 16u; ++index) {
        if (word(index) != subtitle_word(ocr, index - 1u)) return false;
      }
      for (std::size_t index = subtitle_contour_history_reserved_offset; index < subtitle_contour_history_word_count; ++index) {
        if (word(index) != 0u) return false;
      }
      const bool current = owner_frame == completed.matched_frame_id;
      const auto clock = subtitle_u64(identity, 20u);
      if (owner_frame == 0u || clock == 0u ||
          clock != subtitle_u64(locator, subtitle_plane_observation_timestamp_offset) ||
          (current ? clock != completed.matched_observation_timestamp_us :
            (!completed.gpu_undecided_completion || owner_frame >= completed.matched_frame_id ||
             clock > completed.matched_observation_timestamp_us || clock != owner_clock))) return false;
      frame published = completed;
      published.matched_frame_id = owner_frame;
      published.matched_observation_timestamp_us = clock;
      if (!subtitle_records_match_frame(ocr, locator, published)) return false;
      const bool snapshot_valid = source_width <= subtitle_contour_source_max_extent &&
                                  source_height <= subtitle_contour_source_max_extent &&
                                  state[frame_valid] == 1.0f && clock == owner_clock &&
                                  subtitle_word(ocr, 2u) == 1u;
      if (word(subtitle_contour_history_snapshot_valid_word) != static_cast<std::uint32_t>(snapshot_valid) ||
          word(3u) != 0u || geometry[control_reserved_0] != 0u) return false;
      const auto plane = subtitle_plane_words(locator);
      // The caller authenticates the complete sealed camera state before this artifact join.
      // Its ready publication and actual current OCR verdict must grant every current slot,
      // including valid empty. An invalid Base probe only suppresses conditioning below.
      const bool geometry_ready = state[frame_valid] == 1.0f &&
        geometry[renderer_authorization_bits] == contract_tag &&
        geometry[joint_plane_mode_bits] == adaptive_policy_id && clock == owner_clock;
      const bool expected_current_valid = geometry_ready && subtitle_word(ocr, 2u) == 1u;
      if ((plane[subtitle_plane_flags_word] == subtitle_plane_current_valid_flag) != expected_current_valid) return false;
      const bool pin_authority = snapshot_valid && subtitle_word(ocr, 4u) != 0u &&
        plane[subtitle_plane_flags_word] == subtitle_plane_current_valid_flag &&
        models::host_sbs_subtitle_plane::current_count(plane) == subtitle_word(ocr, 4u) &&
        plane[subtitle_plane_adaptive_offset] == 1u && plane[subtitle_plane_adaptive_offset + 15u] == 0u;
      return word(2u) == static_cast<std::uint32_t>(pin_authority);
    }

    bool subtitle_contour_clear_policy_is_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<float> &state,
      const std::vector<std::uint8_t> &support,
      const std::vector<std::uint8_t> &field_pin,
      const std::vector<std::uint8_t> &distance,
      const std::vector<std::uint8_t> &history,
      const std::vector<std::uint8_t> &source,
      const std::vector<std::uint8_t> &probability
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      if (identity.size() != subtitle_contour_history_word_count * sizeof(std::uint32_t) ||
          ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t) ||
          state.size() != state_float_count || support.empty() || field_pin.empty() ||
          distance.empty() || history.empty() || source.empty() || probability.empty()) return false;
      const auto zero = [](const auto &bytes) {
        return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t value) { return value == 0u; });
      };
      const bool supported = subtitle_word(identity, 8u) <= subtitle_contour_source_max_extent &&
                             subtitle_word(identity, 9u) <= subtitle_contour_source_max_extent;
      const auto field_width = subtitle_word(identity, 12u);
      const auto field_height = subtitle_word(identity, 13u);
      const auto native_width = supported ? subtitle_word(identity, 8u) : 1u;
      const auto native_height = supported ? subtitle_word(identity, 11u) : 1u;
      if (native_width == 0u || native_height == 0u ||
          native_height > subtitle_contour_source_max_extent ||
          field_width == 0u || field_height == 0u ||
          field_width > subtitle_contour_source_max_extent ||
          field_height > subtitle_contour_source_max_extent) return false;
      const auto native_pixels = static_cast<std::size_t>(native_width) * native_height;
      const auto field_pixels = static_cast<std::size_t>(field_width) * field_height;
      if (support.size() != native_pixels * sizeof(std::uint32_t) ||
          field_pin.size() != field_pixels * sizeof(std::uint32_t) ||
          distance.size() != field_pixels * sizeof(std::uint32_t) ||
          history.size() != native_pixels * 4u * sizeof(float) ||
          source.size() != native_pixels * 4u * sizeof(std::uint16_t) ||
          probability.size() != subtitle_ocr_output_width * subtitle_ocr_output_height * sizeof(float)) return false;
      const auto contour_valid = subtitle_word(identity, 2u);
      if (contour_valid > 1u || (!supported && contour_valid != 0u)) return false;
      if (contour_valid == 0u) {
        if (!zero(support) || !zero(field_pin) ||
            distance.size() % sizeof(std::uint32_t) != 0u) return false;
        for (std::size_t word = 0u; word < distance.size() / sizeof(std::uint32_t); ++word) {
          if (subtitle_word(distance, word) != subtitle_contour_distance_empty_bits) return false;
        }
      }
      const auto geometry_clock = static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(state[gain_last_observation_low])) |
        (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(state[gain_last_observation_high])) << 32u);
      const bool snapshot_eligible = supported && subtitle_word(ocr, 2u) == 1u &&
                                     state[frame_valid] == 1.0f &&
                                     subtitle_u64(identity, 20u) == geometry_clock;
      if (subtitle_word(identity, subtitle_contour_history_snapshot_valid_word) !=
          static_cast<std::uint32_t>(snapshot_eligible)) return false;
      if (!snapshot_eligible) return zero(source) && zero(probability) && zero(history);
      const auto source_height = subtitle_word(identity, 9u);
      const auto crop_top = subtitle_word(identity, 10u);
      const auto content_left = subtitle_word(identity, 16u), content_top = subtitle_word(identity, 17u);
      const auto content_right = subtitle_word(identity, 18u), content_bottom = subtitle_word(identity, 19u);
      const auto current_count = subtitle_word(ocr, 4u);
      if (source_height == 0u || crop_top > source_height || native_height > source_height - crop_top ||
          content_left >= content_right || content_top >= content_bottom ||
          content_right > field_width || content_bottom > field_height ||
          current_count > subtitle_ocr_final_box_capacity) return false;
      const auto covered = [&](const std::size_t pixel) {
        const auto x = pixel % native_width, y = pixel / native_width;
        const auto sx = 2ull * content_left * native_width +
          (2ull * x + 1u) * (content_right - content_left);
        const auto sy = 2ull * content_top * source_height +
          (2ull * (y + crop_top) + 1u) * (content_bottom - content_top);
        for (std::uint32_t slot = 0u; slot < current_count; ++slot) {
          const auto offset = subtitle_ocr_final_box_offset + slot * subtitle_ocr_box_word_count;
          auto search_top = subtitle_word(ocr, offset + 1u);
          auto search_bottom = subtitle_word(ocr, offset + 3u);
          if (sx >= 2ull * subtitle_word(ocr, offset) * native_width &&
              sx < 2ull * subtitle_word(ocr, offset + 2u) * native_width &&
              sy >= 2ull * search_top * source_height &&
              sy < 2ull * search_bottom * source_height) return true;
        }
        return false;
      };
      const auto current_probability = [&](const std::size_t pixel, float &sample) {
        const float ratio_x = static_cast<float>(subtitle_ocr_output_width) / static_cast<float>(native_width);
        const float ratio_y = static_cast<float>(subtitle_ocr_output_height) / static_cast<float>(native_height);
        const float centered_x = static_cast<float>(pixel % native_width) + .5f;
        const float centered_y = static_cast<float>(pixel / native_width) + .5f;
        // Volatile product stores retain the contracted FP32 rounding boundaries even when
        // a future compiler target enables FMA contraction. No threshold epsilon is used.
        const volatile float scaled_x = centered_x * ratio_x;
        const volatile float scaled_y = centered_y * ratio_y;
        const float x = scaled_x - .5f, y = scaled_y - .5f;
        const auto lx = static_cast<int>(std::floor(x)), ly = static_cast<int>(std::floor(y));
        const auto ax = std::clamp(lx, 0, static_cast<int>(subtitle_ocr_output_width) - 1);
        const auto ay = std::clamp(ly, 0, static_cast<int>(subtitle_ocr_output_height) - 1);
        const auto bx = std::clamp(lx + 1, 0, static_cast<int>(subtitle_ocr_output_width) - 1);
        const auto by = std::clamp(ly + 1, 0, static_cast<int>(subtitle_ocr_output_height) - 1);
        const auto p = [&](const int px, const int py) {
          return std::bit_cast<float>(subtitle_word(probability,
            static_cast<std::size_t>(py) * subtitle_ocr_output_width + px));
        };
        const std::array values {p(ax, ay), p(bx, ay), p(ax, by), p(bx, by)};
        if (!std::all_of(values.begin(), values.end(), [](const float value) {
              return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
            })) return false;
        const float fx = x - static_cast<float>(lx), fy = y - static_cast<float>(ly);
        const float upper_delta = values[1u] - values[0u], lower_delta = values[3u] - values[2u];
        const volatile float upper_product = upper_delta * fx;
        const volatile float lower_product = lower_delta * fx;
        const float upper = values[0u] + upper_product, lower = values[2u] + lower_product;
        const float vertical_delta = lower - upper;
        const volatile float vertical_product = vertical_delta * fy;
        sample = upper + vertical_product;
        return true;
      };
      using namespace models::host_sbs_subtitle_glyph;
      std::optional<std::int32_t> current_reference_center;
      const auto scl = [&](const std::size_t index) {
        return subtitle_word(ocr, subtitle_ocr_classifier_history_offset + index);
      };
      const auto scl_u64 = [&](const std::size_t index) {
        return subtitle_u64(ocr, subtitle_ocr_classifier_history_offset + index);
      };
      bool have_classifier_payload = false;
      for (std::size_t index = 0u; index < subtitle_ocr_classifier_history_word_count; ++index)
        have_classifier_payload = have_classifier_payload || scl(index) != 0u;
      if (have_classifier_payload) {
        // This local artifact authenticator proves payload/current identity and bounded current
        // reference geometry. The complete dump reader separately joins that reference to the
        // completed video/screen region; learned votes never become a pixel-search reference.
        const auto center = std::bit_cast<std::int32_t>(scl(20u));
        const auto search_top = std::bit_cast<std::int32_t>(scl(22u));
        const auto trusted_height = std::bit_cast<float>(scl(27u));
        if (subtitle_word(ocr, 2u) != 1u ||
            scl(0u) != subtitle_ocr_classifier_history_tag ||
            scl(1u) != subtitle_ocr_classifier_history_schema ||
            scl(2u) != subtitle_ocr_classifier_history_known_flags ||
            scl(3u) > subtitle_ocr_classifier_prior_cap ||
            scl_u64(4u) == 0u || scl_u64(4u) != subtitle_u64(identity, 20u) ||
            scl_u64(6u) != subtitle_u64(ocr, 5u) || scl_u64(6u) != subtitle_u64(identity, 4u) ||
            scl_u64(8u) != subtitle_u64(ocr, 7u) || scl_u64(8u) != subtitle_u64(identity, 6u) ||
            scl(10u) != native_width || scl(11u) != source_height ||
            scl(12u) != crop_top || scl(13u) != native_height ||
            scl(14u) != field_width || scl(15u) != field_height ||
            scl(16u) != content_left || scl(17u) != content_top ||
            scl(18u) != content_right || scl(19u) != content_bottom ||
            scl(21u) == 0u || scl(21u) > subtitle_contour_source_max_extent ||
            static_cast<std::int64_t>(center) < -static_cast<std::int64_t>(scl(21u)) ||
            static_cast<std::int64_t>(center) > static_cast<std::int64_t>(scl(21u)) ||
            static_cast<std::int64_t>(search_top) < -static_cast<std::int64_t>(subtitle_contour_source_max_extent) ||
            static_cast<std::int64_t>(search_top) > static_cast<std::int64_t>(subtitle_contour_source_max_extent) ||
            scl(23u) > 1u || scl(24u) > 1u ||
            (scl(24u) == 0u ? scl_u64(25u) != 0u :
              scl_u64(25u) == 0u || scl_u64(25u) > scl_u64(4u)) ||
            scl(30u) != 0u || scl(31u) != 0u ||
            !std::isfinite(trusted_height) || trusted_height < 0.f ||
            trusted_height > static_cast<float>(subtitle_ocr_safe_row_bottom - subtitle_ocr_safe_row_top) ||
            (trusted_height == 0.f ? scl(28u) != 0u || scl(29u) != 0u :
              scl(28u) < subtitle_ocr_safe_row_top || scl(28u) >= scl(29u) ||
              scl(29u) > subtitle_ocr_safe_row_bottom)) return false;
        bool have_votes = false;
        for (std::size_t bin = 0u; bin < subtitle_ocr_classifier_prior_bin_count; ++bin) {
          if (scl(32u + bin) > subtitle_ocr_classifier_prior_cap) return false;
          have_votes = have_votes || scl(32u + bin) != 0u;
        }
        if (have_votes && (trusted_height <= 0.f || scl(3u) == 0u)) return false;
        current_reference_center = center;
      }
      std::vector<pair_t> pairs;
      for (std::uint32_t slot = 0u; slot < current_count; ++slot) {
        pair_t pair {};
        for (std::size_t lane = 0u; lane < 4u; ++lane) {
          pair.core[lane] = subtitle_word(ocr, subtitle_ocr_raw_box_offset + slot * subtitle_ocr_box_word_count + lane);
          pair.cover[lane] = subtitle_word(ocr, subtitle_ocr_final_box_offset + slot * subtitle_ocr_box_word_count + lane);
        }
        pair.ribbon = (subtitle_word(ocr, subtitle_ocr_final_box_offset +
          slot * subtitle_ocr_box_word_count + 5u) & subtitle_ocr_box_flag_ribbon) != 0u;
        pairs.push_back(pair);
      }
      const auto glyph = current_support(native_width, native_height, source_height, crop_top,
        rect_t {content_left, content_top, content_right, content_bottom}, pairs,
        [&](const std::size_t pixel) {
          std::array<std::uint16_t, 4u> half {};
          std::memcpy(half.data(), source.data() + pixel * sizeof(half), sizeof(half));
          sample_t sample {};
          sample.valid = half_to_float(half[3u]) == 1.f;
          for (std::size_t lane = 0u; lane < 3u; ++lane) {
            sample.rgb[lane] = half_to_float(half[lane]);
            sample.valid = sample.valid && std::isfinite(sample.rgb[lane]) && sample.rgb[lane] >= 0.f && sample.rgb[lane] <= 1.f;
          }
          return sample;
        }, current_probability, current_reference_center);
      // Authenticate diagnostic RGB history and current detector support independently.
      // Current evidence grants support immediately; retained RGB never replaces that evidence.
      for (std::size_t pixel = 0u; pixel < native_pixels; ++pixel) {
        const auto mask = subtitle_word(support, pixel);
        if (mask > 1u) return false;
        std::array<std::uint16_t, 4u> half {};
        std::memcpy(half.data(), source.data() + pixel * sizeof(half), sizeof(half));
        std::array<float, 4u> color {};
        for (std::size_t lane = 0u; lane < 4u; ++lane) {
          color[lane] = half_to_float(half[lane]);
          if (!std::isfinite(color[lane])) return false;
        }
        if (color[3u] != 0.0f && color[3u] != 1.0f) return false;
        if (color[3u] == 0.0f && std::any_of(half.begin(), half.end(), [](auto bits) { return bits != 0u; })) return false;
        const bool color_valid = color[3u] == 1.0f &&
          std::all_of(color.begin(), color.begin() + 3u, [](float value) { return value >= 0.0f && value <= 1.0f; });
        if (mask != 0u && (!color_valid || contour_valid == 0u)) return false;
        const auto code_bits = subtitle_word(history, pixel * 4u + 3u);
        std::uint32_t count = 0u;
        bool known_code = !color_valid && code_bits == 0u;
        if (color_valid) {
          for (std::uint32_t candidate = 0u; candidate <= subtitle_contour_history_confirmation_observations; ++candidate) {
            const auto code = subtitle_contour_history_feature_valid_bit |
              (candidate << subtitle_contour_history_feature_count_shift) |
              (candidate == subtitle_contour_history_confirmation_observations ? subtitle_contour_history_feature_mask_bit : 0u);
            if (code_bits == std::bit_cast<std::uint32_t>(static_cast<float>(code))) {
              count = candidate;
              known_code = true;
              break;
            }
          }
        }
        float sampled_probability = 0.0f;
        const bool candidate = contour_valid != 0u && color_valid && covered(pixel) &&
          current_probability(pixel, sampled_probability);
        if (!known_code || (!candidate && count != 0u) ||
            mask != static_cast<std::uint32_t>(candidate && count != 0u && glyph[pixel] != 0u)) return false;
        for (std::size_t lane = 0u; lane < 3u; ++lane) {
          const auto anchor_bits = subtitle_word(history, pixel * 4u + lane);
          if (!color_valid || count <= 1u) {
            const auto expected = color_valid ? std::bit_cast<std::uint32_t>(color[lane]) : 0u;
            if (anchor_bits != expected) return false;
          } else {
            const auto anchor = std::bit_cast<float>(anchor_bits);
            if (!std::isfinite(anchor) || anchor < 0.0f || anchor > 1.0f ||
                std::abs(anchor - color[lane]) > subtitle_contour_history_rgb_distance) return false;
          }
        }
      }
      // Preserve model-fault probability bits; they cannot grant a count or support.
      return true;
    }

    bool subtitle_gaussian_fields_are_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &distance,
      const std::vector<std::uint8_t> &weight,
      const std::vector<std::uint8_t> &blend,
      const std::vector<std::uint8_t> &vertical,
      const std::vector<std::uint8_t> &control,
      const std::vector<std::uint8_t> &base,
      const std::vector<std::uint8_t> &final
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      if (identity.size() != subtitle_contour_history_word_count * sizeof(std::uint32_t) ||
          locator.size() != subtitle_plane_state_word_count * sizeof(std::uint32_t) ||
          ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t) ||
          control.size() != sizeof(std::uint32_t)) return false;
      const auto width = subtitle_word(identity, 12u), height = subtitle_word(identity, 13u);
      const auto left = subtitle_word(identity, 16u), top = subtitle_word(identity, 17u);
      const auto right = subtitle_word(identity, 18u), bottom = subtitle_word(identity, 19u);
      if (width == 0u || height == 0u || width > subtitle_contour_source_max_extent ||
          height > subtitle_contour_source_max_extent || left >= right || top >= bottom ||
          right > width || bottom > height) return false;
      const auto bytes = static_cast<std::size_t>(width) * height * sizeof(float);
      if (distance.size() != bytes || weight.size() != bytes || blend.size() != bytes ||
          vertical.size() != bytes || base.size() != bytes || final.size() != bytes) return false;
      // The compatibility vertical artifact is an exact proposal copy. Subtitle transitions
      // have no vertical envelope; the ordinary Base producer still owns its original one.
      if (vertical != blend) return false;
      const auto f = [](const auto &payload, const std::size_t index) {
        return std::bit_cast<float>(subtitle_word(payload, index));
      };
      const auto enabled = subtitle_word(control, 0u);
      const auto plane = subtitle_plane_words(locator);
      if (!models::host_sbs_subtitle_plane::state_words_are_canonical(plane, display_budget())) return false;
      // The legacy distance payload contains the minimum rectangle completion budget D in
      // parallax pixels. Current native support owns it; covers/history cannot create a budget.
      const auto presentation_budget = [&](const std::uint32_t x, const std::uint32_t y) {
        return f(distance, static_cast<std::size_t>(y) * width + x);
      };
      bool have_finite_row = false;
      for (auto y = top; y < bottom; ++y) {
        const auto first_budget = presentation_budget(left, y);
        have_finite_row = have_finite_row || (std::isfinite(first_budget) && first_budget >= 0.0f &&
          subtitle_word(distance, static_cast<std::size_t>(y) * width + left) != 0x80000000u);
      }
      const auto target_bits = plane[subtitle_plane_adaptive_offset + 3u];
      const auto target = std::bit_cast<float>(target_bits);
      const bool authority = subtitle_word(identity, 2u) == 1u &&
        plane[subtitle_plane_flags_word] == subtitle_plane_current_valid_flag &&
        models::host_sbs_subtitle_plane::current_count(plane) != 0u &&
        plane[subtitle_plane_adaptive_offset] == 1u && plane[subtitle_plane_adaptive_offset + 15u] == 0u &&
        std::isfinite(target) && target >= 0.0f && target <= direct_container_limit;
      if (enabled > 1u || enabled != static_cast<std::uint32_t>(
            authority && have_finite_row)) return false;
      const auto source_width = subtitle_word(identity, 8u), source_height = subtitle_word(identity, 9u);
      if (source_width == 0u || source_height == 0u || source_width > subtitle_contour_source_max_extent ||
          source_height > subtitle_contour_source_max_extent) return false;
      // SM5 permits representational round-direction differences in each separate arithmetic
      // operation. Enclose those operations; endpoints remain exact bit-identity branches.
      const auto floor_float = [](const double value) {
        // SM5 arithmetic may flush a positive subnormal operation result to +0.
        if (value > 0.0 && value < std::numeric_limits<float>::min()) return 0.0f;
        float rounded = static_cast<float>(value);
        if (static_cast<double>(rounded) > value) rounded = std::nextafter(rounded, -std::numeric_limits<float>::infinity());
        return rounded;
      };
      const auto ceil_float = [](const double value) {
        // Negative subnormal operation results may similarly flush to -0.
        if (value < 0.0 && value > -std::numeric_limits<float>::min()) return -0.0f;
        float rounded = static_cast<float>(value);
        if (static_cast<double>(rounded) < value) rounded = std::nextafter(rounded, std::numeric_limits<float>::infinity());
        return rounded;
      };
      // D3D11.3 section 3.1.3.2 permits division as reciprocal (1 ULP) followed by
      // a separately rounded multiply (0.5 ULP). Enclose that implementation as well
      // as direct division; no visual error allowance is added.
      // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
      const double reciprocal = 1.0 / static_cast<double>(source_width);
      int reciprocal_exponent = 0;
      std::frexp(reciprocal, &reciprocal_exponent);
      const double reciprocal_ulp = std::ldexp(1.0, reciprocal_exponent - 24);
      const float reciprocal_low = ceil_float(reciprocal - reciprocal_ulp);
      const float reciprocal_high = floor_float(reciprocal + reciprocal_ulp);
      for (std::uint32_t y = 0u; y < height; ++y) {
        const auto row_begin = static_cast<std::size_t>(y) * width * sizeof(float);
        const bool row_changed = !std::equal(blend.begin() + row_begin,
          blend.begin() + row_begin + static_cast<std::size_t>(width) * sizeof(float), base.begin() + row_begin);
        for (std::uint32_t x = 0u; x < width; ++x) {
          const auto index = static_cast<std::size_t>(y) * width + x;
          const auto sx = std::clamp(x, left, right - 1u), sy = std::clamp(y, top, bottom - 1u);
          const auto clamped_index = static_cast<std::size_t>(sy) * width + sx;
          const auto budget = presentation_budget(sx, sy);
          const bool local_budget_valid = std::isfinite(budget) && budget >= 0.0f &&
            subtitle_word(distance, clamped_index) != 0x80000000u;
          const auto source_index = enabled != 0u && local_budget_valid ? clamped_index : index;
          const auto raw_budget = f(distance, index);
          const bool in_content = x >= left && x < right && y >= top && y < bottom;
          if ((raw_budget == 0.0f && subtitle_word(distance, index) != 0u) ||
              (in_content && !(std::isfinite(raw_budget) && raw_budget >= 0.0f) &&
               subtitle_word(distance, index) != subtitle_contour_distance_empty_bits) ||
              (!in_content && subtitle_word(distance, index) != subtitle_contour_distance_empty_bits)) return false;
          const auto b = f(base, source_index), m = f(weight, index), proposal = f(blend, index);
          const auto v = f(vertical, index), result = f(final, index);
          if (!std::isfinite(b) || !std::isfinite(m) || m < 0.0f || m > 1.0f ||
              (m == 0.0f && subtitle_word(weight, index) != 0u) ||
              !std::isfinite(proposal) || !std::isfinite(v) || !std::isfinite(result) ||
              std::abs(b) > direct_container_limit || std::abs(proposal) > direct_container_limit ||
              std::abs(v) > direct_container_limit || std::abs(result) > direct_container_limit) return false;
          if (enabled == 0u) {
            if (subtitle_word(weight, index) != 0u ||
                subtitle_word(blend, index) != subtitle_word(base, index) ||
                subtitle_word(vertical, index) != subtitle_word(base, index) ||
                subtitle_word(final, index) != subtitle_word(base, index)) return false;
            continue;
          }
          if (!row_changed && subtitle_word(final, index) != subtitle_word(base, index)) return false;
          if (!local_budget_valid) {
            // +Inf is a legal empty local row even while another row has current support.
            // The conditioner copies original Base, including nonreplicated padding bits.
            if (subtitle_word(weight, index) != 0u ||
                subtitle_word(blend, index) != subtitle_word(base, index)) return false;
            continue;
          }
          // Match the separate precise shader operations. A zero budget pins the exact UI bits;
          // a feasible immutable Base has exactly zero weight. Rounded interior M endpoints
          // select the same exact copy branches, while intermediate M retains its admission bound.
          const volatile float difference = target - b;
          const volatile float gap_pixels = std::abs(difference) * static_cast<float>(source_width);
          const bool exact_ui = budget == 0.0f;
          const bool exact_base = !exact_ui && (gap_pixels == 0.0f || budget >= gap_pixels);
          float expected_weight = 0.0f;
          if (exact_ui) {
            expected_weight = 1.0f;
          } else if (!exact_base) {
            const volatile float remaining = budget / gap_pixels;
            const volatile float complement = 1.0f - remaining;
            expected_weight = std::clamp(static_cast<float>(complement), 0.0f, 1.0f);
          }
          if ((exact_ui && subtitle_word(weight, index) != std::bit_cast<std::uint32_t>(1.0f)) ||
              (exact_base && subtitle_word(weight, index) != 0u) ||
              std::abs(static_cast<double>(m) - expected_weight) > subtitle_contour_gaussian_weight_absolute_error_limit) return false;
          if (m == 0.0f) {
            if (subtitle_word(blend, index) != subtitle_word(base, source_index)) return false;
          } else if (m == 1.0f) {
            if (subtitle_word(blend, index) != target_bits) return false;
          } else {
            // The proposal is direct budget completion, not a lerp reconstructed from rounded
            // M. Enclose each SM5 divide and UI add/subtract without adding a visual tolerance.
            const auto normalized_exact = static_cast<double>(budget) / static_cast<double>(source_width);
            auto normalized_low = floor_float(normalized_exact);
            auto normalized_high = ceil_float(normalized_exact);
            if (source_width != 1u) {
              const double product_low_exact = static_cast<double>(budget) * reciprocal_low;
              const double product_high_exact = static_cast<double>(budget) * reciprocal_high;
              const float product_low = product_low_exact > 0.0 &&
                  product_low_exact < std::numeric_limits<float>::min() ?
                0.0f : static_cast<float>(product_low_exact);
              const float product_high = static_cast<float>(product_high_exact);
              normalized_low = std::min(normalized_low, product_low);
              normalized_high = std::max(normalized_high, product_high);
            }
            const auto raw_low = difference > 0.0f ?
              floor_float(static_cast<double>(target) - normalized_high) :
              floor_float(static_cast<double>(target) + normalized_low);
            const auto raw_high = difference > 0.0f ?
              ceil_float(static_cast<double>(target) - normalized_low) :
              ceil_float(static_cast<double>(target) + normalized_high);
            // A direct divide/add can round one ULP beyond an endpoint near D==gap. The
            // producer closes that interval with explicit endpoint copies, not a new tolerance.
            const auto endpoint_low = std::min(b, target), endpoint_high = std::max(b, target);
            const auto low = std::clamp(raw_low, endpoint_low, endpoint_high);
            const auto high = std::clamp(raw_high, endpoint_low, endpoint_high);
            if (proposal < low || proposal > high ||
                (proposal == b && subtitle_word(blend, index) != subtitle_word(base, source_index)) ||
                (proposal == target && subtitle_word(blend, index) != target_bits) ||
                (proposal == 0.0f && !std::signbit(b) && !std::signbit(target) &&
                 std::signbit(proposal))) return false;
          }
        }
      }
      // Exact D minimum and the changed-row horizontal Q30 recurrence are independently replayed by the
      // versioned dump reader; this bounded native guard does not invent a second renderer.
      return true;
    }

    bool subtitle_analytic_tuple_is_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &support,
      const std::vector<std::uint8_t> &source,
      const std::vector<std::uint8_t> &matte,
      const std::vector<std::uint8_t> &condition
    ) noexcept {
      using namespace models::depth_coordinate_v2;
      if (identity.size() != subtitle_contour_history_word_count * sizeof(std::uint32_t) ||
          locator.size() != subtitle_plane_state_word_count * sizeof(std::uint32_t) ||
          ocr.size() != subtitle_ocr_record_word_count * sizeof(std::uint32_t) ||
          condition.size() != models::depth_coordinate_v2::subtitle_condition_param_word_count * sizeof(std::uint32_t) ||
          support.size() != matte.size() || source.size() != 2u * matte.size()) return false;
      const auto plane = subtitle_plane_words(locator);
      models::host_sbs_subtitle_plane::condition_words_t params {};
      std::memcpy(params.data(), condition.data(), condition.size());
      if (!models::host_sbs_subtitle_plane::condition_words_match_plane(params, plane, true)) return false;
      std::uint32_t kinds = 0u;
      for (std::uint32_t slot = 0u; slot < models::host_sbs_subtitle_plane::current_count(plane); ++slot) {
        if ((subtitle_word(ocr, subtitle_ocr_final_box_offset + slot * subtitle_ocr_box_word_count + 5u) &
              subtitle_ocr_box_flag_ribbon) != 0u) kinds |= 1u << slot;
      }
      if (params[subtitle_condition_param_active_word] != 0u &&
          (params[subtitle_condition_param_current_kind_mask_word] != kinds || subtitle_word(identity, 2u) != 1u)) return false;
      const bool eligible = subtitle_word(identity, 8u) <= subtitle_contour_source_max_extent &&
                            subtitle_word(identity, 9u) <= subtitle_contour_source_max_extent;
      const auto width = eligible ? subtitle_word(identity, 8u) : 1u;
      const auto height = eligible ? subtitle_word(identity, 11u) : 1u;
      const auto pixels = static_cast<std::size_t>(width) * height;
      if (width == 0u || height == 0u || support.size() != pixels * sizeof(std::uint32_t)) return false;
      if (subtitle_word(identity, 2u) != 1u)
        return std::all_of(matte.begin(), matte.end(), [](auto value) { return value == 0u; });
      std::vector<std::uint32_t> seeds(pixels);
      std::memcpy(seeds.data(), support.data(), support.size());
      const auto expected = models::host_sbs_subtitle_glyph::dense_matte(width, height, seeds,
        [&](const std::size_t index) {
          std::array<std::uint16_t, 4u> bits {};
          std::memcpy(bits.data(), source.data() + index * sizeof(bits), sizeof(bits));
          models::host_sbs_subtitle_glyph::sample_t sample;
          sample.valid = half_to_float(bits[3u]) == 1.0f;
          for (std::size_t lane = 0u; lane < 3u; ++lane) sample.rgb[lane] = half_to_float(bits[lane]);
          return sample;
        });
      if (expected.size() != pixels) return false;
      for (std::size_t pixel = 0u; pixel < pixels; ++pixel) {
        if (seeds[pixel] > 1u || subtitle_word(matte, pixel) != expected[pixel]) return false;
      }
      return true;
    }

    bool subtitle_capture_resources_are_authenticated(const frame &completed) noexcept {
      const bool ocr_present = completed.ocr_box_record != nullptr;
      const bool locator_present = completed.subtitle_locator_state != nullptr;
      if (ocr_present != locator_present) {
        return false;
      }
      const bool active =
        !completed.subtitle_work_suppressed && ocr_present && locator_present;
      return !active ||
             (completed.shadow_base_final_parallax &&
              completed.subtitle_contour_support &&
              completed.subtitle_glyph_matte &&
              completed.subtitle_condition_params &&
              completed.subtitle_contour_field_pin &&
              completed.subtitle_contour_distance &&
              completed.subtitle_contour_identity &&
              completed.subtitle_contour_history &&
              completed.subtitle_contour_source &&
              completed.subtitle_ocr_probability &&
              completed.subtitle_gaussian_weight &&
              completed.subtitle_gaussian_blend &&
              completed.subtitle_gaussian_vertical_conditioned &&
              completed.subtitle_gaussian_control &&
              completed.model_width > 0 && completed.model_height > 0 &&
              models::depth_coordinate_v2::subtitle_ocr_field_is_calibrated(
                static_cast<std::uint32_t>(completed.model_width),
                static_cast<std::uint32_t>(completed.model_height)
              ));
    }

  }  // namespace detail
}  // namespace platf::sbs_debug
