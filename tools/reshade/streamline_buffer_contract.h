// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_buffer_types.h"
#include "streamline_camera_version.h"

#include <algorithm>
#include <array>
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
    // The exact version, or an inexact family's lowest candidate.
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
    // An inexact 2.x family's candidate versions (versioning::family; more
    // than one): a raw tag is canonical only where every candidate's header
    // gives it the same meaning, and the contract is never surveyed.
    std::array<versioning::number, 4> candidates{};
    unsigned candidate_count{};
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

  // The contract of an inexact 2.x family: every candidate's header must agree.
  // Its fields are the agreed ones (the lowest last type; UIAlpha only when
  // every candidate numbers it alike), for diagnostics.
  inline contract for_versions(const versioning::candidates &family) {
    if (family.count == 1) return for_version(family.values[0]);
    contract out;
    if (!family.count) return out;
    out.known = out.ui_color_and_alpha = true;
    for (unsigned i = 0; i != family.count; ++i) {
      const auto &v = family.values[i];
      const auto each = for_version(v);
      out.known = out.known && each.known;
      out.ui_color_and_alpha = out.ui_color_and_alpha && each.ui_color_and_alpha;
      if (!i || !at_least(v, out.version.major, out.version.minor, out.version.patch)) out.version = v;
      out.last_type = i ? std::min(out.last_type, each.last_type) : each.last_type;
      if (i && out.raw_ui_alpha != each.raw_ui_alpha) out.raw_ui_alpha = unknown;
      else if (!i) out.raw_ui_alpha = each.raw_ui_alpha;
    }
    out.surveyed = false;
    out.candidates = family.values;
    out.candidate_count = family.count;
    return out;
  }

  // Raw SDK tag -> canonical number, or unknown when this header did not declare
  // the raw value with a meaning Game 3D can consume.
  inline std::uint32_t canonical(const contract &c, std::uint32_t raw) {
    if (c.candidate_count > 1) {
      if (!c.known) return unknown;
      const auto agreed = canonical(for_version(c.candidates[0]), raw);
      for (unsigned i = 1; i != c.candidate_count; ++i)
        if (canonical(for_version(c.candidates[i]), raw) != agreed) return unknown;
      return agreed;
    }
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

  // Process-wide contract of the hooked interposer, written once when hooks
  // are installed. Readers copy it under a shared lock (an inexact family's
  // candidates do not fit one atomic word).
  namespace detail {
    inline SRWLOCK active_lock = SRWLOCK_INIT;
    inline contract active_value;
  }
  inline void set_active_contract(const contract &c) {
    AcquireSRWLockExclusive(&detail::active_lock);
    detail::active_value = c;
    ReleaseSRWLockExclusive(&detail::active_lock);
  }
  inline void set_active(const versioning::number &v) { set_active_contract(for_version(v)); }
  inline void clear_active() { set_active_contract(contract{}); }
  inline contract active() {
    AcquireSRWLockShared(&detail::active_lock);
    const auto out = detail::active_value;
    ReleaseSRWLockShared(&detail::active_lock);
    return out;
  }
}
