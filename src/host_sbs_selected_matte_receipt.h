/**
 * @file src/host_sbs_selected_matte_receipt.h
 * @brief CPU storage layout of the selected-matte receipt shared by both selector algorithms.
 */
#pragma once

#include "generated/depth_coordinate_v2_contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace models::selected_matte_receipt_v1 {
  inline constexpr std::uint32_t tag = 0x31534d53u;
  inline constexpr std::uint32_t schema = 1u;
  inline constexpr std::uint32_t legacy_algorithm = 1u;
  inline constexpr std::uint32_t current_outline_algorithm = 2u;

  using contour_constants_t = std::array<std::uint32_t, 24u>;
  using contour_identity_t = std::array<std::uint32_t, depth_coordinate_v2::subtitle_contour_history_word_count>;

  // This names storage and the byte-copy boundary only. Consumers independently authenticate
  // the actual GPU receipt, source/SCT/state/clock and draw roles before granting authority.
  struct payload_t {
    std::array<std::uint32_t, 4u> header;
    contour_constants_t contour_constants;
    contour_identity_t contour_identity;
    std::array<std::uint32_t, 2u> transaction_token;
    std::uint32_t algorithm;
    std::uint32_t reserved;
  };

  inline constexpr std::size_t word_bytes = sizeof(std::uint32_t);
  inline constexpr std::size_t byte_count = sizeof(payload_t);
  inline constexpr std::size_t word_count = byte_count / word_bytes;
  inline constexpr std::size_t contour_byte_offset = offsetof(payload_t, contour_constants);
  inline constexpr std::size_t contour_byte_count = sizeof(contour_constants_t);
  inline constexpr std::size_t contour_byte_end = contour_byte_offset + contour_byte_count;
  using words_t = std::array<std::uint32_t, word_count>;

  static_assert(std::is_standard_layout_v<payload_t> && std::is_trivially_copyable_v<payload_t>);
  static_assert(word_bytes == 4u && word_count == 64u && byte_count == 256u);
  static_assert(contour_byte_offset == 16u && contour_byte_count == 96u && contour_byte_end == 112u);
  static_assert(offsetof(payload_t, contour_identity) == 28u * word_bytes);
  static_assert(offsetof(payload_t, transaction_token) == 60u * word_bytes);
  static_assert(offsetof(payload_t, algorithm) == 62u * word_bytes);
  static_assert(offsetof(payload_t, reserved) == 63u * word_bytes);
  static_assert(sizeof(words_t) == sizeof(payload_t));
}  // namespace models::selected_matte_receipt_v1
