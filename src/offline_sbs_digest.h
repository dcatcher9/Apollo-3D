/** @file Lower-case hexadecimal SHA-256, as the offline worker and its job manager exchange it. */
#pragma once

#include "crypto.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace offline_sbs {
  [[nodiscard]] inline std::string sha256_hex(const std::string_view bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    const auto digest = crypto::hash(bytes);
    std::string result(digest.size() * 2, '\0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
      result[index * 2] = digits[digest[index] >> 4u];
      result[index * 2 + 1] = digits[digest[index] & 0x0fu];
    }
    return result;
  }
}  // namespace offline_sbs
