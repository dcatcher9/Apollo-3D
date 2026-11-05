#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <windows.h>

namespace platf::dxgi::detail {
  /** Only a pointer inside the presentation sink needs relocation. Minimize movement over the
   * complete remaining monitor union; a distant virtual source is not a privileged destination.
   * Half-open Windows desktop rectangles keep every selected point outside the sink, including
   * monitors touching its right/bottom edges and malformed overlapping topology observations.
   */
  inline std::optional<POINT> nearest_interactive_pointer_point(POINT pointer, RECT sink, std::span<const RECT> monitors) {
    const auto valid = [](RECT rect) {
      return rect.left < rect.right && rect.top < rect.bottom;
    };
    const auto contains = [](RECT rect, POINT point) {
      return point.x >= rect.left && point.x < rect.right && point.y >= rect.top && point.y < rect.bottom;
    };
    if (!valid(sink) || !contains(sink, pointer)) {
      return std::nullopt;
    }
    std::optional<POINT> nearest;
    long double distance = std::numeric_limits<long double>::max();
    const auto consider = [&](RECT rect) {
      if (!valid(rect)) {
        return;
      }
      POINT candidate {
        std::clamp(pointer.x, rect.left, rect.right - 1),
        std::clamp(pointer.y, rect.top, rect.bottom - 1),
      };
      const auto dx = static_cast<long double>(candidate.x) - pointer.x;
      const auto dy = static_cast<long double>(candidate.y) - pointer.y;
      const auto candidate_distance = dx * dx + dy * dy;
      if (candidate_distance < distance) {
        distance = candidate_distance;
        nearest = candidate;
      }
    };
    for (const auto &monitor : monitors) {
      if (!valid(monitor) || (monitor.left == sink.left && monitor.top == sink.top && monitor.right == sink.right && monitor.bottom == sink.bottom)) {
        continue;
      }
      // Subtract the sink from overlapping rectangles as well: a candidate must always land on
      // an interactive pixel even during a transient overlapping topology observation.
      const auto overlap_left = std::max(monitor.left, sink.left);
      const auto overlap_right = std::min(monitor.right, sink.right);
      const auto overlap_top = std::max(monitor.top, sink.top);
      const auto overlap_bottom = std::min(monitor.bottom, sink.bottom);
      if (overlap_left >= overlap_right || overlap_top >= overlap_bottom) {
        consider(monitor);
        continue;
      }
      consider({monitor.left, monitor.top, overlap_left, monitor.bottom});
      consider({overlap_right, monitor.top, monitor.right, monitor.bottom});
      consider({overlap_left, monitor.top, overlap_right, overlap_top});
      consider({overlap_left, overlap_bottom, overlap_right, monitor.bottom});
    }
    return nearest;
  }
}  // namespace platf::dxgi::detail
