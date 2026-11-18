/**
 * @file src/host_sbs_contour_history_readback.h
 * @brief Exact CPU encoding adapter for native contour-history readback.
 * The caller authenticates physical texture layout and exact completion identity first.
 * The adapter preserves each input lane's value, sign, and NaN payload.
 */
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace models::contour_history_readback {

  // Integer decoding preserves signed zero, binary16 subnormals, infinities, and NaN payloads.
  inline constexpr std::uint32_t binary16_to_binary32_bits(const std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    const std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    if (exponent == 0u) {
      if (mantissa == 0u) return sign;
      int binary32_exponent = 127 - 14;
      while ((mantissa & 0x0400u) == 0u) {
        mantissa <<= 1u;
        --binary32_exponent;
      }
      return sign | (static_cast<std::uint32_t>(binary32_exponent) << 23u) |
             ((mantissa & 0x03ffu) << 13u);
    }
    if (exponent == 0x1fu) return sign | 0x7f800000u | (mantissa << 13u);
    return sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
  }

  inline constexpr float half_to_float(const std::uint16_t value) noexcept {
    return std::bit_cast<float>(binary16_to_binary32_bits(value));
  }

  // Input and output are distinct owned rows, excluding any GPU RowPitch padding.
  // Expand a tightly packed little-endian RGBA16F row into little-endian RGBA32F.
  // Destination storage is supplied by the caller so a writer can reuse one row.
  inline bool expand_rgba16f_row_to_rgba32f(
    const std::span<const std::uint8_t> source,
    const std::span<std::uint8_t> destination
  ) noexcept {
    if (source.size() % (4u * sizeof(std::uint16_t)) != 0u ||
        source.size() > std::numeric_limits<std::size_t>::max() / 2u ||
        destination.size() != source.size() * 2u) return false;
    for (std::size_t offset = 0u; offset < source.size(); offset += 2u) {
      const auto half = static_cast<std::uint16_t>(source[offset]) |
                        static_cast<std::uint16_t>(source[offset + 1u]) << 8u;
      const auto bits = binary16_to_binary32_bits(static_cast<std::uint16_t>(half));
      const auto output = offset * 2u;
      destination[output] = static_cast<std::uint8_t>(bits);
      destination[output + 1u] = static_cast<std::uint8_t>(bits >> 8u);
      destination[output + 2u] = static_cast<std::uint8_t>(bits >> 16u);
      destination[output + 3u] = static_cast<std::uint8_t>(bits >> 24u);
    }
    return true;
  }

  static_assert(binary16_to_binary32_bits(0x0000u) == 0x00000000u);
  static_assert(binary16_to_binary32_bits(0x8000u) == 0x80000000u);
  static_assert(binary16_to_binary32_bits(0x0001u) == 0x33800000u);
  static_assert(binary16_to_binary32_bits(0x3c00u) == 0x3f800000u);
  static_assert(binary16_to_binary32_bits(0x4cc0u) == 0x41980000u);  // feature code 19

}  // namespace models::contour_history_readback
