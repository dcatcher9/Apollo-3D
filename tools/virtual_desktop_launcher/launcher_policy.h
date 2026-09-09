/**
 * @file tools/virtual_desktop_launcher/launcher_policy.h
 * @brief Pure window geometry for the native Windows window router.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

namespace desktop_launcher {
  struct rectangle_t {
    std::int64_t left;
    std::int64_t top;
    std::int64_t right;
    std::int64_t bottom;

    bool operator==(const rectangle_t &) const = default;
  };

  constexpr bool contains(const rectangle_t &outer, const rectangle_t &inner) {
    return inner.left >= outer.left && inner.top >= outer.top &&
           inner.right <= outer.right && inner.bottom <= outer.bottom &&
           inner.right > inner.left && inner.bottom > inner.top;
  }

  /** Center a new window within the target work area, reducing oversize dimensions if needed. */
  constexpr std::optional<rectangle_t> place_in_work_area(
    const rectangle_t &window,
    const rectangle_t &work
  ) {
    if (window.right <= window.left || window.bottom <= window.top || work.right <= work.left || work.bottom <= work.top) {
      return std::nullopt;
    }
    const auto width = std::min(window.right - window.left, work.right - work.left);
    const auto height = std::min(window.bottom - window.top, work.bottom - work.top);
    const auto left = work.left + (work.right - work.left - width) / 2;
    const auto top = work.top + (work.bottom - work.top - height) / 2;
    return rectangle_t {left, top, left + width, top + height};
  }

  /** WINDOWPLACEMENT uses workspace coordinates, including on negative-coordinate monitors. */
  constexpr rectangle_t screen_to_workspace(
    const rectangle_t &screen,
    const rectangle_t &monitor,
    const rectangle_t &work
  ) {
    const auto x_offset = work.left - monitor.left;
    const auto y_offset = work.top - monitor.top;
    return {
      screen.left - x_offset,
      screen.top - y_offset,
      screen.right - x_offset,
      screen.bottom - y_offset,
    };
  }

  constexpr rectangle_t workspace_to_screen(
    const rectangle_t &workspace,
    const rectangle_t &monitor,
    const rectangle_t &work
  ) {
    const auto x_offset = work.left - monitor.left;
    const auto y_offset = work.top - monitor.top;
    return {
      workspace.left + x_offset,
      workspace.top + y_offset,
      workspace.right + x_offset,
      workspace.bottom + y_offset,
    };
  }
}  // namespace desktop_launcher
