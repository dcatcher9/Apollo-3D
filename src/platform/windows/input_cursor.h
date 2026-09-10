/** Cursor geometry shared by Windows exclusive-display clipping and deterministic tests. */
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

namespace platf::detail {
  struct cursor_bounds_t {
    int x, y, width, height;

    bool operator==(const cursor_bounds_t &) const = default;

    [[nodiscard]] bool valid() const {
      return width > 0 && height > 0;
    }
  };

  [[nodiscard]] inline std::optional<cursor_bounds_t> cursor_bounds_intersection(const cursor_bounds_t &first, const cursor_bounds_t &second) {
    if (!first.valid() || !second.valid()) {
      return std::nullopt;
    }
    const auto left = std::max(first.x, second.x);
    const auto top = std::max(first.y, second.y);
    const auto right = std::min<std::int64_t>(static_cast<std::int64_t>(first.x) + first.width, static_cast<std::int64_t>(second.x) + second.width);
    const auto bottom = std::min<std::int64_t>(static_cast<std::int64_t>(first.y) + first.height, static_cast<std::int64_t>(second.y) + second.height);
    if (right <= left || bottom <= top) {
      return std::nullopt;
    }
    return cursor_bounds_t {left, top, static_cast<int>(right - left), static_cast<int>(bottom - top)};
  }

}  // namespace platf::detail
