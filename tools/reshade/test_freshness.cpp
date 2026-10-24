// SPDX-License-Identifier: GPL-3.0-only
// The named time limits of freshness.h (refactor plan P2): each limit's value
// and, for each rule its call sites keep, the edge one millisecond either side
// of it, a tick ahead of the clock and a zero tick. The seam characterization
// drives the capture owner's and the provider's sites; the Streamline probe
// reads its own clock, so this table is the pin of the limits and edges its
// sites call.
#include "freshness.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace {
  using namespace sunshine_freshness;

  unsigned failures {};

  enum class rule { fresh_lt, fresh_le, ahead_tolerant_lt, ahead_tolerant_le, within_source_age };

  bool fresh(rule edge, std::uint64_t tick, std::uint64_t now, std::uint64_t limit) {
    switch (edge) {
      case rule::fresh_lt: return fresh_lt(tick, now, limit);
      case rule::fresh_le: return fresh_le(tick, now, limit);
      case rule::ahead_tolerant_lt: return ahead_tolerant_lt(tick, now, limit);
      case rule::ahead_tolerant_le: return ahead_tolerant_le(tick, now, limit);
      case rule::within_source_age: return within_source_age(tick, now);
    }
    return false;
  }

  // One limit as its call sites read it.
  struct row {
    const char *pins;
    std::uint64_t limit, value;
    rule edge;
  };

  constexpr row rows[] {
    {"Selection, completed snapshots, the delivering owner, the display hold and UI requests keep the source age below it, a zero tick unavailable",
     source_age_ms, 250, rule::within_source_age},
    {"Active evaluation views, a pending nomination, a readable capture and the probe's recent camera keep the source age below it",
     source_age_ms, 250, rule::fresh_lt},
    {"The exporter holds a projection scene at most the source age", source_age_ms, 250, rule::fresh_le},
    {"A Streamline tag's recorded copy keeps precedence over DLSS-G's input below the source age, also ahead of the clock",
     source_age_ms, 250, rule::ahead_tolerant_lt},
    {"An owner's evaluations keep its association at most the association limit", association_ms, 1000, rule::fresh_le},
    {"An FG scope's enable, its accepted copies and DLSS-G's evaluation supply stay live at most the FG silence, also ahead of the clock",
     fg_silent_ms, 1000, rule::ahead_tolerant_le},
    {"A provider handoff holds, on an evaluation as recent, at most the handoff limit", handoff_ms, 250, rule::fresh_le},
    {"A frame's camera serves a depth tag at most the camera age older", camera_age_ms, 250, rule::fresh_le},
    {"The probe's evaluation, depth source and presentation queries admit evidence at most the evidence age old",
     evidence_age_ms, 250, rule::fresh_le},
    {"The probe's UI scope query admits a tag below the evidence age", evidence_age_ms, 250, rule::fresh_lt},
  };

  // Fresh at limit - 1, limit and limit + 1 ms, and one ms ahead of the clock.
  struct outcome {
    bool below, at, past, ahead;
  };

  outcome expected(rule edge) {
    switch (edge) {
      case rule::fresh_lt:
      case rule::within_source_age: return {true, false, false, false};
      case rule::fresh_le: return {true, true, false, false};
      case rule::ahead_tolerant_lt: return {true, false, false, true};
      case rule::ahead_tolerant_le: return {true, true, false, true};
    }
    return {};
  }

  void check(bool condition, const char *pins, const char *what) {
    if (!condition) {
      ++failures;
      std::fprintf(stderr, "FAIL %s: %s\n", pins, what);
    }
  }
}

int main() {
  constexpr std::uint64_t now = 100000;
  for (const auto &r : rows) {
    check(r.limit == r.value, r.pins, "the limit changed");
    const auto want = expected(r.edge);
    check(fresh(r.edge, now - (r.limit - 1), now, r.limit) == want.below, r.pins, "one ms below the limit");
    check(fresh(r.edge, now - r.limit, now, r.limit) == want.at, r.pins, "at the limit");
    check(fresh(r.edge, now - (r.limit + 1), now, r.limit) == want.past, r.pins, "one ms past the limit");
    check(fresh(r.edge, now + 1, now, r.limit) == want.ahead, r.pins, "one ms ahead of the clock");
    // A tick far ahead never wraps into an old one.
    check(fresh(r.edge, UINT64_MAX, 0, r.limit) == want.ahead, r.pins, "a tick far ahead of the clock");
    check(fresh(r.edge, now, now, r.limit), r.pins, "a tick of this millisecond");
  }
  check(!within_source_age(0, 0) && !within_source_age(0, source_age_ms - 1),
    "The source age's rule", "a zero tick is unavailable, never age zero");
  check(fresh_lt(0, source_age_ms - 1, source_age_ms) && fresh_le(0, source_age_ms, source_age_ms),
    "The comparators", "a zero tick is an ordinary tick; its sites reject it themselves");
  if (failures) {
    std::fprintf(stderr, "Freshness limits: %u cells changed\n", failures);
    return 1;
  }
  std::printf("PASS freshness limits: %zu rows, each limit's value and edge\n", std::size(rows));
  return 0;
}
