/** Session-local cursor policy shared by Windows injection and deterministic tests. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

namespace platf::detail {
  using cursor_point_t = std::pair<int, int>;

  /** A cloned GDI source cannot provide a cursor destination private to this monitor. */
  class cursor_source_identity_t {
  public:
    void observe(bool active_interface, bool expected_identity) {
      if (active_interface) {
        ++active_interfaces_;
        expected_identity_ = expected_identity_ || expected_identity;
      }
    }

    [[nodiscard]] bool private_target(bool enumeration_complete) const {
      return enumeration_complete && active_interfaces_ == 1 && expected_identity_;
    }

  private:
    unsigned active_interfaces_ = 0;
    bool expected_identity_ = false;
  };

  struct cursor_bounds_t {
    int x, y, width, height;

    [[nodiscard]] bool valid() const {
      return width > 0 && height > 0;
    }

    [[nodiscard]] bool contains(cursor_point_t point) const {
      return valid() && point.first >= x && point.second >= y &&
             static_cast<std::int64_t>(point.first) < static_cast<std::int64_t>(x) + width &&
             static_cast<std::int64_t>(point.second) < static_cast<std::int64_t>(y) + height;
    }

    [[nodiscard]] cursor_point_t clamp(std::int64_t px, std::int64_t py) const {
      return {
        static_cast<int>(std::clamp<std::int64_t>(px, x, static_cast<std::int64_t>(x) + width - 1)),
        static_cast<int>(std::clamp<std::int64_t>(py, y, static_cast<std::int64_t>(y) + height - 1))
      };
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

  /** No OS-wide lock: only a new remote action may restore this session's saved position. */
  class confined_cursor_state_t {
  public:
    [[nodiscard]] std::optional<cursor_point_t> move(
      const std::optional<cursor_bounds_t> &bounds,
      const std::optional<cursor_point_t> &initial_position,
      int dx,
      int dy
    ) {
      if (!active_ || !bounds || !bounds->valid()) {
        return std::nullopt;
      }
      if (!position_) {
        position_ = initial_position && bounds->contains(*initial_position) ?
                      *initial_position :
                      cursor_point_t {bounds->x + bounds->width / 2, bounds->y + bounds->height / 2};
      }
      const auto start = bounds->clamp(position_->first, position_->second);
      position_ = bounds->clamp(static_cast<std::int64_t>(start.first) + dx, static_cast<std::int64_t>(start.second) + dy);
      return position_;
    }

    [[nodiscard]] std::optional<cursor_point_t> absolute(
      const std::optional<cursor_bounds_t> &bounds,
      float x,
      float y
    ) {
      if (!active_ || !bounds || !bounds->valid() || !std::isfinite(x) || !std::isfinite(y)) {
        return std::nullopt;
      }
      position_ = bounds->clamp(
        static_cast<std::int64_t>(bounds->x) + static_cast<std::int64_t>(std::clamp(x, 0.0f, 1.0f) * bounds->width),
        static_cast<std::int64_t>(bounds->y) + static_cast<std::int64_t>(std::clamp(y, 0.0f, 1.0f) * bounds->height)
      );
      return position_;
    }

    void observe_game_cursor(cursor_point_t position) {
      if (active_) {
        position_ = position;
      }
    }

    void reset() {
      active_ = false;
      position_.reset();
    }

  private:
    bool active_ = true;
    std::optional<cursor_point_t> position_;
  };

  /** Pixel-center encoding avoids rounding a monitor's last pixel onto its neighbour. */
  [[nodiscard]] inline int cursor_pixel_to_absolute(int pixel, int origin, int extent) {
    if (extent <= 0) {
      return 0;
    }
    const auto relative = std::clamp<std::int64_t>(static_cast<std::int64_t>(pixel) - origin, 0, extent - 1);
    return static_cast<int>(std::min<std::int64_t>(65535, (relative * 65536 + 32768) / extent));
  }

  [[nodiscard]] inline bool game_cursor_already_confined(
    const cursor_bounds_t &monitor,
    const cursor_bounds_t &clip,
    cursor_point_t cursor,
    bool cursor_hidden
  ) {
    return cursor_hidden && clip.valid() && monitor.contains(cursor) &&
           monitor.contains({clip.x, clip.y}) &&
           monitor.contains({clip.x + clip.width - 1, clip.y + clip.height - 1});
  }
}  // namespace platf::detail
