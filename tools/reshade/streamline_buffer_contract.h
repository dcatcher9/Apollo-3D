// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_buffer_types.h"
#include "streamline_camera_version.h"

#include <atomic>
#include <cstdint>

// Streamline buffer-type numbers are header constants compiled into the game.
// They are not stable across SDK releases: 1.x tag 23 is UIHint, not 2.x
// UIColorAndAlpha, and 2.11.x tag 68 is UIAlpha while 2.12.0 renumbered it to
// 69 and reused 68 for ResponsivityMask. Every raw tag is translated once, at
// the hook boundary, into the canonical numbering of the newest surveyed public
// header. Consumers then compare canonical numbers only.
//
// Survey: NVIDIA-RTX/Streamline include/sl.h (1.0.0-2.4.15) and
// include/sl_core_types.h (2.7.2-2.14.1). Types are append-only between the
// surveyed releases except for the two changes above.
namespace sunshine_streamline::buffers {
  struct contract {
    versioning::number version;
    // Highest raw type declared by the governing header; later values are unknown.
    std::uint32_t last_type{};
    // Raw number of UIAlpha in this header, or unknown when it was not declared.
    std::uint32_t raw_ui_alpha{unknown};
    // False for 1.x, whose tag 23 is UIHint rather than UIColorAndAlpha.
    bool ui_color_and_alpha{};
    // The version lies inside the surveyed public range. Newer releases keep
    // the latest surveyed numbering but are not treated as authenticated.
    bool surveyed{};
    bool known{};
  };

  inline bool at_least(const versioning::number &v, unsigned major, unsigned minor, unsigned patch = 0) {
    if (v.major != major) return v.major > major;
    if (v.minor != minor) return v.minor > minor;
    return v.patch >= patch;
  }

  // Contract for an identified interposer version. Unknown majors have no contract.
  inline contract for_version(const versioning::number &v) {
    contract out;
    out.version = v;
    if (v.major == 1) {
      // 1.0.0-1.0.4 end at Position (32); 1.1.x adds InvalidDepthMotionHint (33).
      out.last_type = at_least(v, 1, 1) ? 33 : 32;
      out.surveyed = !at_least(v, 1, 2);
      out.known = true;
      return out;
    }
    if (v.major != 2) return out;
    out.known = out.ui_color_and_alpha = true;
    out.surveyed = !at_least(v, 2, 15);
    if (at_least(v, 2, 12)) { out.last_type = 72; out.raw_ui_alpha = 69; }
    else if (at_least(v, 2, 11)) { out.last_type = 68; out.raw_ui_alpha = 68; }
    else if (at_least(v, 2, 7, 30)) out.last_type = 67;
    else if (at_least(v, 2, 7)) out.last_type = 66;
    else if (at_least(v, 2, 4)) out.last_type = 53;
    else out.last_type = 37;
    return out;
  }

  // Raw SDK tag -> canonical number, or unknown when this header did not declare
  // the raw value with a meaning Game 3D can consume.
  inline std::uint32_t canonical(const contract &c, std::uint32_t raw) {
    if (!c.known || raw > c.last_type) return unknown;
    if (raw == ui_color_and_alpha && !c.ui_color_and_alpha) return unknown;
    if (c.raw_ui_alpha != unknown && raw == c.raw_ui_alpha) return ui_alpha;
    // 2.11.x declares 68 only as UIAlpha; 2.12+ keeps 68 as ResponsivityMask.
    return raw;
  }

  // A dedicated UI-alpha tag is admissible for automatic protection only when
  // the loaded interposer's own header range assigns it that meaning.
  inline bool ui_alpha_authenticated(const contract &c) {
    return c.known && c.surveyed && c.raw_ui_alpha != unknown;
  }

  // Process-wide contract of the hooked interposer. Written once when hooks are
  // installed; readers on SDK threads take a relaxed snapshot of the packed value.
  namespace detail {
    inline std::atomic<std::uint64_t> active_version{};
    inline std::uint64_t pack(const versioning::number &v) {
      return (std::uint64_t(v.major & 0x7fff) << 48) | (std::uint64_t(v.minor & 0xffff) << 32) |
        (std::uint64_t(v.patch & 0xffff) << 16) | std::uint64_t(v.build & 0xffff);
    }
  }
  inline void set_active(const versioning::number &v) {
    detail::active_version.store(detail::pack(v) | 1ull << 63, std::memory_order_release);
  }
  inline void clear_active() { detail::active_version.store(0, std::memory_order_release); }
  inline contract active() {
    const auto packed = detail::active_version.load(std::memory_order_acquire);
    if (!(packed >> 63)) return {};
    const versioning::number v{unsigned(packed >> 48 & 0x7fff), unsigned(packed >> 32 & 0xffff),
      unsigned(packed >> 16 & 0xffff), unsigned(packed & 0xffff)};
    return for_version(v);
  }
}
