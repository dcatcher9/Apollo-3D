/** Ownership-safe Windows cursor clipping for an exclusive virtual/AR display layout. */
#pragma once

#include "input_cursor.h"

#include <algorithm>
#include <cwctype>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace platf::primary_display::detail {
  using cursor_bounds_t = platf::detail::cursor_bounds_t;

  struct cursor_clip_io_t {
    std::function<std::optional<cursor_bounds_t>()> query;
    std::function<std::optional<cursor_bounds_t>()> desktop;
    std::function<bool(std::optional<cursor_bounds_t>)> apply;
  };

  class cursor_clip_manager_t {
  public:
    explicit cursor_clip_manager_t(cursor_clip_io_t io):
        io_(std::move(io)) {}

    bool confine(std::wstring_view identity, const cursor_bounds_t &bounds) {
      if (identity.empty() || !bounds.valid() || (owned_ && !same_identity(identity))) {
        return false;
      }
      const auto current = io_.query();
      if (!current) {
        return false;
      }
      if (!owned_ || *current != *owned_) {
        // Another application's replacement becomes the state to preserve. Never restore an
        // older clip over it. Keep a compatible application restriction, but a disjoint clip
        // must be overridden or it can trap the shared cursor on another active display.
        owned_.reset();
        previous_ = current;
        const auto desktop = io_.desktop();
        if (!desktop) {
          return false;
        }
        previous_unbounded_ = *current == *desktop;
      }
      const auto overlap = previous_unbounded_ ? std::optional<cursor_bounds_t> {} :
                                                 platf::detail::cursor_bounds_intersection(bounds, *previous_);
      const auto desired = overlap ? *overlap : bounds;
      // Foreground notifications are intentionally redundant. Avoid calling ClipCursor when
      // Windows still reports the exact restriction that this process already owns.
      if (owned_ && *current == *owned_ && desired == *owned_) {
        return true;
      }
      if (!io_.apply(desired)) {
        return false;
      }
      // Keep tentative ownership on a failed verification query so cleanup can retry safely.
      owned_ = desired;
      identity_ = identity;
      const auto observed = io_.query();
      if (!observed) {
        return false;
      }
      if (*observed != *owned_) {
        owned_.reset();
        previous_.reset();
        return false;
      }
      return true;
    }

    bool release(std::wstring_view expected_identity = {}) {
      if (owned_ && !expected_identity.empty() && !same_identity(expected_identity)) {
        return false;
      }
      if (owned_) {
        const auto current = io_.query();
        if (!current) {
          return false;
        }
        if (*current == *owned_ && !io_.apply(previous_unbounded_ ? std::nullopt : previous_)) {
          return false;
        }
      }
      owned_.reset();
      previous_.reset();
      previous_unbounded_ = false;
      identity_.clear();
      return true;
    }

    bool owns_clip() const {
      return owned_.has_value();
    }

  private:
    bool same_identity(std::wstring_view identity) const {
      return identity.size() == identity_.size() && std::equal(identity.begin(), identity.end(), identity_.begin(), [](wchar_t a, wchar_t b) {
               return std::towupper(a) == std::towupper(b);
             });
    }

    cursor_clip_io_t io_;
    std::optional<cursor_bounds_t> previous_;
    std::optional<cursor_bounds_t> owned_;
    bool previous_unbounded_ = false;
    std::wstring identity_;
  };
}  // namespace platf::primary_display::detail
