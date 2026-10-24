// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>

// How old API depth evidence may be when it is read: each time limit of
// selection, association, the FG scope, the provider handoff, held copies and
// the Streamline probe's camera and query evidence, defined once. Each call
// site names its limit and the comparator that keeps its own edge; the edges
// differ by site and are not unified here. Ticks are GetTickCount64
// milliseconds, and a tick's age at now is now - tick. Cadences, log
// throttles, the capture pool's leases, calibration and UI detection keep
// their constants with their owners.
namespace sunshine_freshness {
  // A source's nomination, its completed snapshots, a held display copy and
  // the projection held for a capture without its own camera. Multi-frame
  // generation can present while two real frames are still in flight.
  inline constexpr std::uint64_t source_age_ms = 250;
  // An established source keeps its queue mono through gaps only while its
  // API still evaluates it. A game that stops calling the API without a
  // release (DLSS -> TAA, a cutscene without DLSS) loses the association
  // after this silence, and Generic takes over through its usual hysteresis.
  // Evaluations the API rejects for as long count as silence: an API that
  // fails every call delivers no depth (BF6 10-22 held 70 s of mono on one).
  inline constexpr std::uint64_t association_ms = 1000;
  // An enabled FG scope with neither a new enable nor an accepted FG copy for
  // this long is silent, and the viewport's ordinary SR/RR depth may supply
  // depth (depth_capture::frame_generation_scope_live). It is the
  // association's second.
  inline constexpr std::uint64_t fg_silent_ms = association_ms;
  // A provider handoff holds API ownership in mono at most this long after
  // the last associated Present, and needs an evaluation this recent.
  inline constexpr std::uint64_t handoff_ms = 250;
  // A frame's camera may be at most this much older than the Streamline
  // depth tag that uses it.
  inline constexpr std::uint64_t camera_age_ms = 250;
  // The Streamline probe's queries' default age limit for the evidence they
  // return (streamline_camera_probe.h).
  inline constexpr std::uint64_t evidence_age_ms = 250;

  // Below the limit (<) or at most the limit (<=). A tick ahead of now is
  // not fresh.
  constexpr bool fresh_lt(std::uint64_t tick, std::uint64_t now, std::uint64_t limit) {
    return now >= tick && now - tick < limit;
  }
  constexpr bool fresh_le(std::uint64_t tick, std::uint64_t now, std::uint64_t limit) {
    return now >= tick && now - tick <= limit;
  }
  // The same edges, where a tick ahead of now (another thread stamped it
  // after this reader read its clock) is fresh.
  constexpr bool ahead_tolerant_lt(std::uint64_t tick, std::uint64_t now, std::uint64_t limit) {
    return now < tick || now - tick < limit;
  }
  constexpr bool ahead_tolerant_le(std::uint64_t tick, std::uint64_t now, std::uint64_t limit) {
    return now < tick || now - tick <= limit;
  }
  // The source age's rule for selection, held copies and UI requests: a
  // nonzero tick younger than the source age. A zero tick is unavailable,
  // never age zero.
  constexpr bool within_source_age(std::uint64_t tick, std::uint64_t now) {
    return tick && fresh_lt(tick, now, source_age_ms);
  }
}
