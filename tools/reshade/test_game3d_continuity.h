// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The continuity invariant (README, Session-level tests), defined once for
// the native provider fixture's frame scripts and the API contract matrix's
// runtime layer, which includes that fixture. The matrix judges single
// decisions; Assassin's Creed Shadows' depth flicker came from correct
// decisions composed over frames, which only a session shows.
//
// For every Present after warm-up: if at least one legal source had depth
// for the frame it presents, the Present shows depth from some source, or
// holds earlier depth within the source age (freshness.h). A Present whose
// frame had no legal source may hold within the source age, then go mono. A
// generated Present, which the game makes while it records a later frame,
// presents an earlier frame: after a Present that showed depth it holds that
// depth while the copy is younger than the source age.
//
// The freshness rule, next to it: a Present that shows API depth shows the
// newest copy of that source the presenting queue could legally read for the
// frame it presents, within freshness_bound_ms of it (two of the fixture's
// frame intervals). The
// source is the shown rung's: an FG scope's own depth (D1: an FG tag's copy
// or DLSS-G's input) or scene depth (D2: an SR or RR evaluation's input, or a
// tag's copy outside FG), never the other, since FG keeps its viewport while
// it can supply (frame_generation_holds) and a switch restarts the placement
// calibration. The supply log says which copies were readable: a legal
// source of the presenting queue's own (one that meets no limitation and
// that no live FG scope keeps out) from its frame's Present (its own
// Present already for a copy recorded at its call, supply::immediate), and a copy
// recorded on another queue once its work completed, if the contract copies
// it out (docs/reshade-sbs.md, Copy-out on the producer queue;
// supply::completions). The record names the shown copy by its capture tick
// (frame_record::depth_tick_ms) and its capture id (depth_capture). Where the
// fixture knows each copy's capture id as well (supply::capture, the session
// timelines), the rule is exact: a newer copy of the shown copy's own kind
// (an FG scope's input, or ordinary depth) among the rung's sources that the
// presenting queue could read is stale_shown with no bound (until 10-28 only
// the tick bound below applied, which a single Present's lag, about one
// frame, always passed). Showing older depth while newer copies were
// readable is stale_shown: no row excuses it, as no known limitation makes a
// readable copy unreadable. After FG Off (supply::fg_off) an FG copy is
// held as the viewport's display lineage's (C8) and judged against the
// lineage's copies of either kind, also those the FG hold kept off-screen
// while FG was on (supply::fg_held): FG Off retires only the FG role, so
// the lineage's newest readable copy replaces the held one (10-28; until
// then an FG hold beside a readable newer SR copy passed). A real Present
// that shows a copy of the other
// API than the last real Present that showed depth shows a copy recorded
// after that one (capture ids follow publication order across providers):
// an ownership handover never steps back to older depth.
//
// A judge reads two things per Present: the frame record the test add-on
// exports (SunshineGame3DTestFrameRecord, docs/reshade-sbs.md, Frame
// record) and the fixture's own supply log, what the frame it presents
// supplied as the fixture knows it (never what the add-on reports). A
// generated Present supplies nothing of its own.
//
// Warm-up: the judge is armed by a Present that shows depth. A Present that
// shows none disarms it when its frame had nothing it could show (no legal
// source its own Present reads and nothing holdable within the source age,
// the mono the invariant permits), when its frame's events may end its
// source (supply::drop_permitted), or when it is a violation; a generated
// Present neither arms nor disarms it. While disarmed, a frame's legal
// sources have one source age to show depth, counted from the first such
// Present, and establish_ms only during a warm-up: the session's start, a
// stall (a Present at least the source age after the previous one: every
// hold expired with no Present in between, and a pipelined source such as
// DLSS-G's ring has nothing readable within the age until it refills; never
// for a frame whose own Present reads one of its legal sources,
// supply::immediate, wherever the pause came, nor for a generated Present
// after its frame's real Present; always after a silence of the association
// window) or a new source (a source-age gap in legal sources). Frames whose sources meet
// a known limitation and frames whose sources meet none are timed apart, so
// a loss a known gap excuses never extends the time the others have.
//
// A violation of a known limitation is a known gap: the row that names the
// limitation records it, and the fixture tags the frames that meet it. A
// row records a loss only once no copy the presenting queue could read (a
// frame's legal source that meets no limitation, and that no live FG scope
// kept out of a frame it supplies: supply::owned) is younger than the source
// age, as its limitation explains no earlier one, so that a hold dropped
// within that age is no row's. A violation no row records fails its run,
// and so does a row whose
// limitation was supplied in a run without a violation (the gap closed: the
// fix removes the row).
//
// Two rows are limitations the owner approved: C2 until Step 4, and
// R1-late-signaller, O1's residual of the queue-order model (a signal first
// submitted after a read that relied on its fence), whose fail-safe drops
// the hold itself: it records a generated Present's dropped hold after such
// a signal and the late reads, while a real Present is judged by what else
// it could hold (judge::late_signal, supply::unordered). C1, API
// depth recorded on a queue other than the presenting one, closed on 10-27:
// every such copy is copied out on its own queue (U-C1b gives a source's
// first copies that storage too) and read once complete or once the game's
// own fence waits order the presenting queue after it (U-C1). A copy whose
// queue's copy-out ring was all in flight (copy_out ring_busy, a resource
// bound of eight lists per queue that no fixture fills) meets no row: a loss
// it caused fails its run. A matrix row's recorded gap over frames would
// also be one (none is open since C5 closed on 10-26). Any other row is a defect these tests found
// (found: when and by which test), excused only until the owner decides it
// (README, Found defects; none is open since C8 closed on 10-26): a strict run
// (SUNSHINE_GAME3D_CONTINUITY_STRICT=1, the contract runner's
// -StrictContinuity) fails their violations as any other. A random session timeline may supply a limitation
// without losing depth, so the rows the session timelines found
// (test_game3d_session_timeline_runtime.cpp) are kept by their witness
// timelines instead, each of which must lose depth on every run. The rows
// follow the API contract matrix's known-gap discipline
// (test_game3d_api_contract_known_gaps.h) with their own key, the
// limitation, since no oracle cell describes a session.
#include "freshness.h"
#include "game3d_frame_record.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace game3d_continuity {
  // The sources a frame can supply, as a mask.
  enum source : std::uint8_t {
    // An SR or RR evaluation (Streamline or NGX) whose depth input's state
    // its recording or declaration proves.
    sr_evaluate = 1,
    // A depth tag whose copy the contract allows: at its call (an FG tag, or
    // OnlyValidNow outside FG scope) or before its frame's Present (R2).
    tag_copy = 2,
    // DLSS-G's own DLSSG.Depth in a state its recording names (N4, N7).
    dlssg_input = 4,
    // A Generic depth-stencil boundary whose state a proof names.
    generic_boundary = 8,
  };

  inline constexpr std::uint8_t api_sources = sr_evaluate | tag_copy | dlssg_input;
  inline constexpr std::uint8_t generic_sources = generic_boundary;
  // The sources of each API rung (the freshness rule): D1, an FG scope's own
  // depth, and D2, scene depth. A tag's copy is either, as FG is on or off.
  inline constexpr std::uint8_t fg_depth_sources = tag_copy | dlssg_input;
  inline constexpr std::uint8_t scene_depth_sources = sr_evaluate | tag_copy;

  // A known limitation the frame's legal source meets; a gap row names it.
  enum class limitation : std::uint8_t {
    none,
    // A Generic depth-stencil view drawn on a queue other than the
    // presenting one, whose only boundary is the end of the frame.
    cross_queue_generic,
    // O1's residual: a generated Present after a real Present that read a
    // copy from another queue in flight, ordered by a fence of the game's
    // that a signal first submitted after that read (the CPU's or another
    // queue's) then reached: the add-on stops holding that copy
    // (late_conflict), so it may drop the hold there.
    late_signaller,
  };

  // What one frame supplied, stamped when the fixture made the first call
  // that supplies it (no later than the add-on's own capture tick).
  struct supply {
    // Sources the contract requires the add-on to be able to show.
    std::uint8_t legal {};
    // Sources it may show but need not: an EITHER outcome, or a Generic view
    // the add-on leaves aside while an API source owns the queue (H).
    std::uint8_t permitted {};
    // The legal sources that meet a known limitation; limit names it when
    // every legal source meets one (the frame is tagged with it).
    std::uint8_t limited {};
    // The legal sources a live FG scope keeps out of the frame it expects to
    // supply, the one after a frame that supplied an FG copy: the Present
    // shows that scope's FG depth instead, so they leave no copy a later
    // Present could hold (no readable copy below), while the frame's own
    // Present is judged as any.
    std::uint8_t owned {};
    // The legal sources recorded on a queue other than the presenting one
    // (the freshness rule): such a copy is readable only through one of
    // completions, never from its frame's Present.
    std::uint8_t elsewhere {};
    // The legal sources whose copy its own Present reads: one recorded at
    // its call on a list the presenting queue executes before that Present
    // (an SR evaluation's input, a tag copied at its call, DLSS-G's input on
    // the game's list), and one from another queue readable there
    // (completions' copy, complete or ordered: U-C1). The freshness rule
    // counts the first kind at that Present (until 10-27 no copy of a frame
    // counted before the Present after it, so a Present that held a copy up
    // to the source age old beside its frame's own fresh copy passed), and a
    // stall inside such a frame, after its first call, is no warm-up: the
    // judge measures the stall rule to the frame's stamp (judge::present).
    // So is a copy deferred to its frame's Present (R2) that no untag
    // cancels: the add-on records it on the presenting queue right before
    // that Present decides (since 10-28; until then it kept the
    // Present-to-Present rule, so a hitch between the frame's calls and its
    // Present passed a mono Present as a stall's warm-up). DLSS-G's ring and
    // an unordered frame's copies keep the Present-to-Present rule.
    std::uint8_t immediate {};
    limitation limit {limitation::none};
    // tick: the frame's first call that supplies (the stall rule's stamp).
    // made: when each source's copy was made, by source bit (sr_evaluate,
    // tag_copy, dlssg_input, generic_boundary): its deciding call, after any
    // hitch before it, its recording's submission for a copy from another
    // queue, or its frame's Present for a copy deferred there (R2, since
    // 10-28: after a hitch before that Present its tag call's tick made the
    // frame's own copy look as old as the hitch). holdable, readable_after and the freshness rule age copies by
    // it (until the 10-27 review by tick, so after a hitch between a frame's
    // first call and its copy a 20 ms old copy looked 420 ms old and a
    // dropped hold of it passed).
    std::uint64_t tick {};
    std::array<std::uint64_t, 4> made {};
    // capture: each source's copy as the add-on names it, by source bit (0:
    // unknown, as for a copy deferred to its frame's Present): its capture
    // id (frame_record::depth_capture), with copy_fg_input set for an FG
    // scope's input. The freshness rule is exact where both are known.
    std::array<std::uint64_t, 4> capture {};
    // The frame supplied an FG copy its live scope accepts: DLSS-G adds its
    // generated Presents, which present it (judge::present, between).
    bool fg_frame {};
    // The frame's events may end its source's hold (an untag, the retiring
    // of its viewport): a mono Present there is no dropped hold.
    bool drop_permitted {};
    // Its Present comes after FG Off (its FG options turned FG off, at the
    // frame's start or after its evaluations): an FG copy it shows is held
    // across FG Off as its display lineage's (C8), so the freshness rule
    // judges it against the lineage's readable copies of either kind, and
    // the copies an FG hold kept off-screen (fg_held) are readable there.
    // The session timelines set it (one viewport each); the native provider
    // fixture's scripts keep the rung's own kind.
    bool fg_off {};
    // The owned sources that the display's hold of a silent FG scope's own
    // copy keeps off-screen while FG is on (frame_generation_holds, an owner
    // rule), unlike the ones a live scope refuses: they are copied, and
    // after FG Off they are the viewport's display lineage's successor
    // copies, readable at a Present of a frame with fg_off (until 10-28 the
    // judge never counted them, so an FG copy held beside them after FG Off
    // passed).
    std::uint8_t fg_held {};
    // A signal after this frame's real Present reached a fence a read in
    // flight relied on (O1): its real Present is judged as any, and a hold
    // its generated Presents drop is R1-late-signaller's where the copy the
    // real Present showed was read through that fence (judge::late_signal;
    // until the 10-27 review any such drop was, also of a copy the
    // presenting queue made itself).
    bool late_after {};
    // Its own Present can read none of its legal sources: each is a copy
    // from another queue that no fence of the game's orders that Present
    // after (U-C1), except SR a live FG scope keeps out (owned). The Present
    // holds earlier depth, and may go mono only once no copy it could read
    // is younger than the source age (judge::holdable); later Presents read
    // these copies once complete (completions).
    bool unordered {};
    // One probe per copy recorded on another queue, with its source: true
    // once its producer's work completed, when the presenting queue can read
    // its copy-out (every such copy is copied out since U-C1b, 10-27). relied:
    // the game's order fence (the fixture's identity of that fence object,
    // new at each FG toggle) through which that copy was readable at its own
    // Present before it completed (U-C1's relied read), else null: a late
    // signal of it makes that copy alone unholdable (judge::late_signal), as
    // O1 stops holding such a read (until the 10-27 review every copy of the
    // frame took the frame's fences, keyed by a struct reused across FG
    // toggles).
    struct completion {
      std::uint8_t sources {};
      std::function<bool()> probe;
      const void *relied {};
      // Its work completed before its frame's Present call
      // (judge::before_present): readable at that Present's decision.
      bool done_before_present {};
      // Its capture id with copy_fg_input (supply::capture), 0 unknown.
      std::uint64_t capture {};
      // An FG hold keeps it off-screen while FG is on (supply::fg_held).
      bool fg_held {};
    };
    std::vector<completion> completions;

    void add(std::uint8_t sources, std::uint64_t now, limitation meets = limitation::none) {
      legal |= sources;
      if (meets != limitation::none) {
        limited |= sources;
        limit = meets;
      }
      if (!tick) {
        tick = now;
      }
      made_at(sources, now);
    }

    void allow(std::uint8_t sources, std::uint64_t now) {
      permitted |= sources;
      if (!tick) {
        tick = now;
      }
      made_at(sources, now);
    }

    // The sources' copies made at now, where not made earlier in the frame.
    void made_at(std::uint8_t sources, std::uint64_t now) {
      for (unsigned i = 0; i != made.size(); ++i) {
        if ((sources & (1u << i)) && !made[i]) {
          made[i] = now;
        }
      }
    }

    // The newest made time of these sources' copies, else the frame's tick.
    std::uint64_t made_tick(std::uint8_t sources) const {
      std::uint64_t out {};
      for (unsigned i = 0; i != made.size(); ++i) {
        if (sources & (1u << i)) {
          out = std::max(out, made[i]);
        }
      }
      return out ? out : tick;
    }

    // A legal copy recorded on another queue; probe: its completion.
    void add_elsewhere(std::uint8_t sources, std::uint64_t now, std::function<bool()> probe = {}) {
      add(sources, now);
      elsewhere |= sources;
      completes(sources, std::move(probe));
    }

    void completes(std::uint8_t sources, std::function<bool()> probe, const void *relied = nullptr, std::uint64_t id = 0, bool held = false) {
      if (probe) {
        completions.push_back({sources, std::move(probe), relied, false, id, held});
      }
    }
  };

  // supply::capture's flag: the copy is an FG scope's input.
  inline constexpr std::uint64_t copy_fg_input = 1ull << 63;

  // How long legal sources may take to show depth after the judge was
  // disarmed: an enabled FG scope may own its viewport for fg_silent_ms with
  // no copy of its own (freshness.h), and the viewport's source then has an
  // association window to be nominated, copied and read.
  inline constexpr std::uint64_t establish_ms = sunshine_freshness::fg_silent_ms + sunshine_freshness::association_ms;
  // Two GetTickCount64 periods: the fixture's stamp and the add-on's tick are
  // separate reads of a clock that advances in steps of about 16 ms.
  inline constexpr std::uint64_t tick_slack_ms = 40;
  // The slack of a fixture's limitation windows (the session timelines'
  // known-gap tags and its other-queue window): each window starts at the
  // fixture's own read of the clock at a copy or an evaluation and ends at
  // its read at a later frame's end, while the add-on decides at its own
  // reads at that frame's Present, after the frame's remaining calls and up
  // to three generated Presents (each about one 16 ms period). Two clock
  // periods and that latency come to about 100 ms; a wider window would
  // excuse losses the limitation does not explain.
  inline constexpr std::uint64_t tag_slack_ms = 100;
  // The freshness rule's bound: how much older than the newest copy the
  // presenting queue could certainly read at the Present (its supply stamp)
  // the shown copy (its capture tick) may be. Certainly readable: a copy of
  // the presenting queue's own from a frame before the one the Present
  // presents (a copy deferred to its frame's Present, R2, is read at the
  // Present after it), or from that frame when recorded at its call
  // (supply::immediate), and a copy from another queue that the judge saw
  // complete at an earlier Present or right before this Present's call (its
  // work completed before this Present's decision). Two of the fixture's frame intervals, its own pacing (the mean
  // of its last two real Present intervals) at whatever rate it runs: the
  // presented frame's own copy and the one before it, which the add-on may
  // show instead. The stamp and the tick are separate reads of a clock that
  // advances in 16 ms steps, so the bound adds tick_slack_ms.
  inline constexpr unsigned freshness_frames = 2;

  constexpr std::uint64_t freshness_bound_ms(std::uint64_t frame_interval_ms) {
    return freshness_frames * frame_interval_ms + tick_slack_ms;
  }

  enum class violation_kind : std::uint8_t {
    // An armed judge: the frame had a legal source and the Present showed
    // no depth.
    gap,
    // Legal sources established no depth within their window (one source
    // age, or establish_ms during a warm-up).
    unestablished,
    // A rung showed pixels at least the source age (and the slack) after
    // the newest legal or permitted source of its kind (API or Generic).
    stale,
    // A generated Present showed no depth after a Present that showed depth,
    // while the fresh copy the last real Present showed was certainly
    // younger than the source age (less the slack): it dropped the hold of
    // the frame it presents.
    dropped_hold,
    // The freshness rule: a Present showed API depth captured more than
    // freshness_bound_ms before the newest copy of its rung's sources that
    // the presenting queue could read by then.
    stale_shown,
  };

  constexpr const char *name(violation_kind value) {
    switch (value) {
      case violation_kind::gap:
        return "gap";
      case violation_kind::unestablished:
        return "unestablished";
      case violation_kind::dropped_hold:
        return "dropped_hold";
      case violation_kind::stale_shown:
        return "stale_shown";
      default:
        return "stale";
    }
  }

  constexpr const char *name(limitation value) {
    switch (value) {
      case limitation::cross_queue_generic:
        return "cross_queue_generic";
      case limitation::late_signaller:
        return "late_signaller";
      default:
        return "none";
    }
  }

  struct violation {
    violation_kind kind {};
    const char *where {""};  // The frame script (or phase) it happened in.
    std::uint64_t present {}, tick {};
    supply frame;
    // stale: since the newest supply of its kind; unestablished: since
    // warm-up began.
    std::uint64_t age_ms {};
    sunshine_game3d::source_slot owner {sunshine_game3d::source_slot::none};
    sunshine_game3d::source_state state {sunshine_game3d::source_state::inactive};
    const char *reason {""};
    // The age of the newest frame that supplied an API source the presenting
    // queue could read (a legal source that meets no limitation), whose copy
    // the Present can hold while it is younger than the source age; none:
    // UINT64_MAX.
    std::uint64_t readable_age_ms {UINT64_MAX};
    // stale_shown: the shown copy's capture tick, the frames that supplied
    // a readable copy after it (age_ms: the newest one's lead) and the bound.
    std::uint64_t shown_tick {}, bound_ms {};
    unsigned newer_frames {};
    // stale_shown by capture id: the shown copy and the newer readable copy
    // (or, at a provider handover, the previous real Present's copy).
    std::uint64_t shown_capture {}, newer_capture {};
  };

  // Known limitations: a violation whose frame's legal source meets a row's
  // limitation is that row's (gap and unestablished; staleness has none).
  struct gap_row {
    const char *id;
    limitation limit;
    // A defect these tests found, awaiting the owner's decision: the date
    // and the test that found it. nullptr: an approved limitation, or a
    // recorded matrix gap over frames.
    const char *found;
    const char *owner_note;
    // Every loss of a tagged frame is the row's, also one within the source
    // age of a readable copy: the limitation drops that hold itself.
    bool drops_hold {};
  };

  inline constexpr gap_row known_gaps[] {
    {"C2-cross-queue-generic", limitation::cross_queue_generic, nullptr, "Until Step 4's fence hooks. Generic's end-of-frame copy runs on the presenting queue, and a proof belongs to the queue whose executed list made it, so a depth-stencil view drawn on another queue with no boundary but the end of the frame is refused (missing_state; 18e5d922, docs/reshade-sbs.md, the last proven state). Seven of the eight games of the 10-06 to 10-23 logs render on another queue. Either owner option (accept the other queue's proof once fence hooks show the presenting queue ordered after it, or copy on the proving queue) shows depth here and removes this row."},
    {"R1-late-signaller", limitation::late_signaller, nullptr, "Owner decision O1 (10-27, Step 4 design section 1): accepted. A Present reads a copy from another queue in flight once the game's own fence waits order the presenting queue after its copy-out (U-C1); no observation at that decision can exclude a signal of the waited fence, first submitted after it by the CPU or another queue, that runs before the producer's own signal. The add-on poisons such a fence for good (O3), counts each relied read it reaches within the source age as late_conflict, and stops holding that read's presentation copy from the next Present (fail-safe: mono rather than pixels its producer may not have written). The session timelines tag the generated Presents of a frame whose fence such a signal reaches after its Present (supply::late_after), and judge real Presents by what else they could hold, a copy read through the signalled fence being unholdable from the signal on (judge::late_signal; until 10-27 a window tagged every legal source of every real frame within the source age, which excused any loss there); unordered_read counts a read the script orders by its calls up to the read's Present, but not with every later signal, as this row's (test_queue_order_script.h, judge_reads). A tagged generated Present's dropped hold is the row's: the fail-safe drops the hold itself.", true},
  };
  inline constexpr std::size_t gap_count = std::size(known_gaps);

  // A strict run: the found defects' rows excuse nothing.
  inline bool strict() {
    static const bool value = [] {
      const char *text = std::getenv("SUNSHINE_GAME3D_CONTINUITY_STRICT");
      return text && *text && std::strcmp(text, "0") != 0;
    }();
    return value;
  }

  constexpr int known(const violation &value) {
    if (value.kind == violation_kind::stale || value.kind == violation_kind::stale_shown || value.frame.limit == limitation::none) {
      return -1;
    }
    for (std::size_t i = 0; i != gap_count; ++i) {
      if (known_gaps[i].limit != value.frame.limit) {
        continue;
      }
      // A limitation loses depth once the presenting queue's newest copy
      // aged out (C1's note); a loss while that copy could still be held is
      // no row's.
      if (!known_gaps[i].drops_hold && value.readable_age_ms < sunshine_freshness::source_age_ms) {
        return -1;
      }
      return int(i);
    }
    return -1;
  }

  // The row that excuses a violation, or -1: known(), except that in a
  // strict run a found defect's row excuses none.
  inline int excused(const violation &value) {
    const auto row = known(value);
    return row >= 0 && strict() && known_gaps[std::size_t(row)].found ? -1 : row;
  }

  inline std::string describe(const violation &value) {
    char text[800];
    std::snprintf(text, sizeof(text), "%s in %s at present=%llu tick=%llu: frame legal=0x%x permitted=0x%x limitation=%s supplied_at=%llu age_ms=%llu readable_age_ms=%lld shown_tick=%llu newer_frames=%u bound_ms=%llu shown_capture=%llu newer_capture=%llu; record owner=%s state=%s reason=%s", name(value.kind), value.where, static_cast<unsigned long long>(value.present), static_cast<unsigned long long>(value.tick), unsigned(value.frame.legal), unsigned(value.frame.permitted), name(value.frame.limit), static_cast<unsigned long long>(value.frame.tick), static_cast<unsigned long long>(value.age_ms), value.readable_age_ms == UINT64_MAX ? -1ll : static_cast<long long>(value.readable_age_ms), static_cast<unsigned long long>(value.shown_tick), value.newer_frames, static_cast<unsigned long long>(value.bound_ms), static_cast<unsigned long long>(value.shown_capture), static_cast<unsigned long long>(value.newer_capture), sunshine_game3d::name(value.owner), sunshine_game3d::name(value.state), value.reason ? value.reason : "");
    return text;
  }

  // Whether a Present's record shows depth: the rung that decided it has
  // pixels (D1 or D2 ready or held, or Generic's binding ready). H in mono
  // and a Present no rung decided show none.
  constexpr bool shows_depth(const sunshine_game3d::frame_record &record) {
    using sunshine_game3d::source_slot;
    using sunshine_game3d::source_state;
    const auto owner = record.depth_owner;
    if (owner != source_slot::fg_depth && owner != source_slot::scene_depth && owner != source_slot::generic) {
      return false;
    }
    const auto state = record[owner].state;
    return state == source_state::ready || state == source_state::held;
  }

  struct row_tally {
    unsigned exercised {}, violations {};
  };

  // The found defects' violations of a run, for its summary: "<row>=<n>"
  // each, and what a strict run does with them; "none" without such a row.
  inline std::string found_defects(const std::array<row_tally, gap_count> &rows) {
    if (std::none_of(std::begin(known_gaps), std::end(known_gaps), [](const gap_row &row) {
          return row.found != nullptr;
        })) {
      return "none";
    }
    std::string out;
    unsigned total {};
    for (std::size_t i = 0; i != gap_count; ++i) {
      if (known_gaps[i].found) {
        out += std::string(out.empty() ? "" : " ") + known_gaps[i].id + "=" + std::to_string(rows[i].violations);
        total += rows[i].violations;
      }
    }
    return out + (total ? strict() ? " (strict: failed)" : " (excused until the owner decides; SUNSHINE_GAME3D_CONTINUITY_STRICT=1 fails them)" : "");
  }

  class judge {
  public:
    using row_tally = game3d_continuity::row_tally;

    // A frame with no legal source must hold the last depth within the
    // source age: its mono Present while a copy is holdable is a dropped
    // hold unless its supply permits the drop (supply::drop_permitted). The
    // session timelines, which mark the events that end a source, set it;
    // the native provider fixture's frame scripts end sources without such
    // marks and keep the older rule (a frame without a legal source may go
    // mono).
    bool strict_holds {};

    // One Present: its frame record and the supply of the frame it presents
    // (empty for a generated Present, which presents the frame the previous
    // real Present presented), in the script or phase where.
    void present(const sunshine_game3d::frame_record &record, const supply &frame, const char *where, bool generated = false, bool between = false) {
      using sunshine_game3d::source_slot;
      ++presents_;
      const auto now = record.tick_ms;
      // After FG Off the copies an FG hold kept off-screen are readable
      // (supply::fg_held); a generated Present supplies nothing of its own.
      fg_off_ = !generated && frame.fg_off;
      // Whether this real Present's frame has a legal source its own Present
      // reads (supply::immediate). Not an owned one: a live FG scope refuses
      // the copy of the SR evaluation it keeps out (streamline_camera_probe
      // .cpp, fg_route::owned), so that frame's own Present can show only the
      // scope's FG copy, judged by what it could hold (the 10-27 review's J2
      // took the owned copy for one its Present reads).
      const bool reads_own = !generated && frame.tick && (frame.immediate & frame.legal & ~frame.limited & ~frame.owned);
      // A generated Present that presents the last real Present's frame as
      // that Present did (between: after it, before the next frame's first
      // depth call; the fixture declares the frame's generated count as the
      // game's numFramesToGenerate): the add-on counts that frame's own copy
      // current there (frame_presenting) when the frame made a
      // frame-generation call, and holds the copy its real Present showed.
      // must_hold: the frame supplied an FG copy its live scope accepts
      // (supply::fg_frame), so the add-on does count it.
      const bool own_generated = generated && between;
      const bool must_hold = own_generated && presented_.fg_frame;
      // A stall: no Present for the source age. A frame whose own Present
      // reads one of its legal sources never stalled, wherever the pause
      // came (before its first call or inside it): that source is readable
      // there however long the pause was (until the 10-27 review a pause
      // before the frame's first call still disarmed the judge, which
      // excused up to establish_ms of mono). Nor does a generated Present of
      // that frame after its real Present: the pause is the frame's, whose
      // real Present was judged. A silence of the association window or more
      // is a warm-up for every frame: the API association ends there by
      // design (freshness.h, association_ms).
      const bool silence = last_present_ && now >= last_present_ + sunshine_freshness::association_ms;
      if (last_present_ && now >= last_present_ + sunshine_freshness::source_age_ms && (silence || !(reads_own || own_generated))) {
        ++stalls_;
        armed_ = false;
        showing_ = false;
        warming_ = true;
        since_ = {};
      }
      last_present_ = now;
      if (!generated) {
        presented_ = frame;
        real_presents_ = {now, real_presents_[0], real_presents_[1]};
        fresh_ = false;
        shown_copy_ = {};
        readable(frame, now);
      }
      if (frame.limited & api_sources) {
        newest_limited_ = frame.tick ? frame.tick : now;
      }
      const auto stamp = frame.tick ? frame.tick : now;
      const auto offered = std::uint8_t(frame.legal | frame.permitted);
      // The newest offered API copy's made tick (supply::made): after a
      // hitch inside the frame before its copy, its stamp is that much
      // older than the copy (until 10-28 the stamp: a Present of a frame with
      // no API source holding the copy of a frame with a brief hitch before
      // its calls was stale at a 157 ms old copy, aged 297 ms).
      if (offered & api_sources) {
        newest_api_ = frame.tick ? frame.made_tick(std::uint8_t(offered & api_sources)) : now;
      }
      if (offered & generic_sources) {
        newest_generic_ = stamp;
      }
      if ((frame.legal & api_sources & ~frame.owned) && frame.limit == limitation::none && !frame.unordered) {
        newest_readable_ = std::max(newest_readable_, frame.made_tick(std::uint8_t(frame.legal & api_sources & ~frame.owned)));
      }
      if (frame.legal) {
        ++legal_presents_;
        for (std::size_t i = 0; i != gap_count; ++i) {
          rows_[i].exercised += known_gaps[i].limit == frame.limit;
        }
      }
      const auto owner = record.depth_owner;
      const auto &row = owner == source_slot::none ? record[source_slot::hold] : record[owner];
      // The freshness rule and holdable read what was certainly readable
      // before this Present's decision (complete at an earlier look, or at
      // before_present), then look at the copies from other queues again: a
      // copy whose work completed between the add-on's decision and this
      // record (its frame's own, or an earlier frame's) was not readable at
      // that decision, so it neither excuses nor demands anything here
      // (until 10-27 holdable read it after that look, so an unordered
      // frame's mono Present was a gap or not by when DLSS-G's copy
      // completed: seed c28965dca8d7c76c, frame 38).
      std::pair<unsigned, std::uint64_t> fresher {};
      std::uint64_t newer_capture {};
      if ((owner == source_slot::fg_depth || owner == source_slot::scene_depth) && record.depth_tick_ms) {
        // An FG copy is held as its display lineage's, FG on or off: a newer
        // readable copy of either kind replaces it (after FG Off,
        // supply::fg_off). While FG is on the SR copies its owner keeps
        // off-screen are not readable here (supply::owned never enters the
        // readable set, and readable_now hides supply::fg_held), so the
        // owner's frame_generation_holds rule stands; an SR copy readable
        // beside the held FG copy (a silent scope whose held copy a late
        // signal conflicted, O1) must replace it. Until 10-28 the lineage was
        // judged after FG Off only, and such a hold passed (the 10-28
        // review's J1).
        const bool lineage = owner == source_slot::fg_depth;
        const auto sources = lineage ? api_sources : owner == source_slot::fg_depth ? fg_depth_sources : scene_depth_sources;
        fresher = readable_after(record.depth_tick_ms, sources);
        newer_capture = readable_newer(record.depth_capture, sources, lineage);
      }
      const bool could_hold = holdable(now);
      look();
      // frame: the supply the violation is judged with (a generated
      // Present's is the frame it presents).
      const auto report = [&](violation_kind kind, std::uint64_t age, const supply &judged) {
        violation value {kind, where, record.present, now, judged, age, owner, row.state, row.reason};
        if (newest_readable_) {
          value.readable_age_ms = now > newest_readable_ ? now - newest_readable_ : 0;
        }
        if (const auto index = known(value); index >= 0) {
          ++rows_[std::size_t(index)].violations;
        }
        pending_.push_back(value);
      };
      if (shows_depth(record)) {
        ++shown_;
        const bool generic = owner == source_slot::generic;
        shown_generic_ += generic;
        const auto [newer, newest_readable] = fresher;
        if (newer_capture) {
          // Exact: a newer copy of the shown copy's kind was readable.
          violation value {violation_kind::stale_shown, where, record.present, now, frame, newest_readable > record.depth_tick_ms ? newest_readable - record.depth_tick_ms : 0, owner, row.state, row.reason};
          value.shown_tick = record.depth_tick_ms;
          value.newer_frames = newer;
          value.shown_capture = record.depth_capture;
          value.newer_capture = newer_capture & ~copy_fg_input;
          pending_.push_back(value);
        } else if (newer && real_presents_[2]) {
          const auto bound = freshness_bound_ms((real_presents_[0] - real_presents_[2]) / 2);
          if (newest_readable - record.depth_tick_ms > bound) {
            violation value {violation_kind::stale_shown, where, record.present, now, frame, newest_readable - record.depth_tick_ms, owner, row.state, row.reason};
            value.shown_tick = record.depth_tick_ms;
            value.newer_frames = newer;
            value.bound_ms = bound;
            pending_.push_back(value);
          }
        }
        // A handover to the other API steps forward: its copy was recorded
        // after the copy the last real Present showed.
        if (!generated && !generic && record.depth_capture) {
          const auto api = record[owner].key.api;
          if (handover_.capture && handover_.api != api && record.depth_capture < handover_.capture) {
            violation value {violation_kind::stale_shown, where, record.present, now, frame, 0, owner, row.state, row.reason};
            value.shown_tick = record.depth_tick_ms;
            value.shown_capture = record.depth_capture;
            value.newer_capture = handover_.capture;
            pending_.push_back(value);
          }
          handover_ = {record.depth_capture, api};
        }
        // The newest source of the shown rung's kind; a tick ahead of the
        // record's (another read of the clock) is age 0. So is a real
        // Present's API copy captured inside the frame it presents (its
        // capture tick no earlier than the frame's stamp, both reads of one
        // clock): that frame's own depth, however long the frame took. Until
        // 10-27 its age ran from the frame's first call to the Present, so a
        // hitch inside the frame before its SR evaluation made the frame's
        // own fresh copy stale (the 10-27 full check's C8-fg-off-color-only,
        // frame 39: age 672 ms after a 683 ms Present gap). So is the copy a
        // frame's real Present showed, at a generated Present of that frame
        // after it (own_generated: frame_presenting). An older copy shown
        // there, or any copy of a frame with no API source, keeps the age
        // since the newest source.
        const auto newest = generic ? newest_generic_ : newest_api_;
        const bool own_frame = !generated && !generic && frame.tick && record.depth_tick_ms >= frame.tick;
        const bool own_generated_copy = own_generated && !generic && record.depth_tick_ms &&
          ((shown_copy_.tick && record.depth_tick_ms == shown_copy_.tick) || (presented_.tick && record.depth_tick_ms >= presented_.tick));
        std::uint64_t age = now;
        if (own_frame || own_generated_copy) {
          age = 0;
        } else if (newest) {
          age = now > newest ? now - newest : 0;
        }
        // The hold's own age: a real Present that holds an API copy older
        // than its frame (not its frame's own) shows it only while that copy
        // is younger than the source age, whatever newer supply its frame
        // made. Until the 10-27 review only the newest supply's age was
        // judged, so after a hitch the next frame's Present could hold the
        // copy before the hitch, 418 ms old, while its own copy was deferred
        // to a later Present (R2), and pass.
        if (!generated && !generic && !own_frame && record.depth_tick_ms && now > record.depth_tick_ms) {
          age = std::max(age, now - record.depth_tick_ms);
        }
        if (!newest || age >= sunshine_freshness::source_age_ms + tick_slack_ms) {
          report(violation_kind::stale, age, frame);
        }
        armed_ = true;
        showing_ = true;
        if (!generated) {
          fresh_ = row.state == sunshine_game3d::source_state::ready && (frame.legal & api_sources) && !frame.limited;
          if (owner == source_slot::fg_depth || owner == source_slot::scene_depth) {
            shown_copy_ = {record.depth_tick_ms, owner == source_slot::fg_depth ? fg_depth_sources : scene_depth_sources};
          }
        }
        warming_ = false;
        since_ = {};
        last_legal_ = frame.legal ? now : last_legal_;
        return;
      }
      const bool held = std::exchange(showing_, false);
      if (generated) {
        // It presents the frame the previous real Present presented and holds
        // the depth shown there. A generated Present of an FG frame after its
        // real Present (own_generated) presents that frame as its real
        // Present did: dropping the frame's own fresh copy the real Present
        // showed is a dropped hold whatever the clock (until the 10-27 review
        // it counted only within the source age of the clock, so after a
        // hitch the add-on's mono there passed: open item 1). Any other
        // generated Present holds that depth while the copy is younger than
        // the source age, less the slack, measured from the shown copy's
        // capture when it is the presented frame's own (a frame whose first
        // call came before a hitch and its copy after it), else from the
        // real Present two before the last (a copy deferred to a later
        // Present, R2, was made after it), unless a copy recorded on another
        // queue (a limited source) may have become readable since. It
        // neither arms nor disarms.
        const bool own_copy = fresh_ && shown_copy_.tick && presented_.tick && shown_copy_.tick >= presented_.tick;
        const bool cross_queue = newest_limited_ && now < newest_limited_ + sunshine_freshness::source_age_ms + tick_slack_ms;
        const auto since = own_copy ? shown_copy_.tick : real_presents_[2];
        const bool within = since && now + tick_slack_ms < since + sunshine_freshness::source_age_ms;
        if (held && fresh_ && !cross_queue && ((must_hold && own_copy) || within)) {
          auto judged = presented_;
          if (judged.late_after && shown_conflicted()) {
            judged.limit = limitation::late_signaller;
          }
          report(violation_kind::dropped_hold, since && now > since ? now - since : 0, judged);
        }
        return;
      }
      // A mono real Present. A frame whose own Present reads none of its
      // legal sources (an unordered frame, a copy deferred to the Present
      // after it, R2, or a pipelined source) and that has nothing it could
      // hold younger than the source age (holdable, by each copy's made
      // time) supplies that Present nothing: it may go mono, as may a frame
      // whose events end its source (supply::drop_permitted). Until the
      // 10-27 review only unordered frames had this rule, and a legal frame
      // whose copy only a later Present reads was a gap after a hitch, where
      // only an over-age hold could show depth. A frame whose sources meet a
      // known limitation keeps its loss: its row records it there (known).
      if (frame.drop_permitted || (!reads_own && !could_hold && !frame.limited)) {
        armed_ = false;
        return;
      }
      if (!frame.legal) {
        // A frame with no legal source holds the last depth within the
        // source age (could_hold): under strict_holds its mono Present is a
        // dropped hold (until the 10-27 review any frame without a legal
        // source could go mono, and its disarm gave the next legal frames a
        // new source's warm-up). A judge without strict_holds keeps that rule
        // for frame scripts that end sources without marking them.
        if (strict_holds && armed_) {
          report(violation_kind::dropped_hold, newest_readable_ && now > newest_readable_ ? now - newest_readable_ : 0, frame);
        }
        armed_ = false;
        return;
      }
      if (armed_) {
        report(violation_kind::gap, 0, frame);
        armed_ = false;
        since(frame) = now;
      } else {
        if (!last_legal_ || now - last_legal_ >= sunshine_freshness::source_age_ms) {
          // A new source: a warm-up, timed from this frame.
          warming_ = true;
          since_ = {};
        }
        auto &start = since(frame);
        if (!start) {
          start = now;
        } else if (now - start > (warming_ ? establish_ms : sunshine_freshness::source_age_ms)) {
          report(violation_kind::unestablished, now - start, frame);
          start = now;
        }
      }
      last_legal_ = now;
    }

    // A fresh judge for the next segment of the same session (a frame
    // script): its arming, warm-up and tallies start again, while the supply
    // history stays, since a Present may hold a copy of the segment before.
    void restart() {
      judge next;
      next.strict_holds = strict_holds;
      next.last_present_ = last_present_;
      next.newest_api_ = newest_api_;
      next.newest_generic_ = newest_generic_;
      next.newest_readable_ = newest_readable_;
      next.newest_limited_ = newest_limited_;
      next.frames_ = frames_;
      next.copies_ = std::move(copies_);
      next.handover_ = handover_;
      *this = std::move(next);
    }

    // The first signal of a game's order fence that the CPU or another queue
    // submitted after reads relied on it (O1): the copies read through it
    // are no longer holdable, as the add-on stops holding such a read
    // (late_conflict), so they neither excuse a loss nor make a shown older
    // copy stale.
    void late_signal(const void *fence) {
      for (auto &copy : copies_) {
        copy.conflicted = copy.conflicted || (fence && copy.relied == fence);
      }
    }

    // Right before a real Present call, after its frame's last submission:
    // the copies from other queues complete now, the frame's own (pending)
    // among them, completed before that Present's decision.
    void before_present(supply &pending) {
      look();
      for (auto &value : pending.completions) {
        value.done_before_present = value.probe && value.probe();
      }
    }

    // A Present with no new frame record (the add-on wrote none): not judged.
    void unrecorded() {
      ++unrecorded_;
    }

    // The violations since the last take.
    std::vector<violation> take() {
      std::vector<violation> out;
      out.swap(pending_);
      return out;
    }

    const std::array<row_tally, gap_count> &rows() const {
      return rows_;
    }

    unsigned presents() const {
      return presents_;
    }

    unsigned legal_presents() const {
      return legal_presents_;
    }

    unsigned shown() const {
      return shown_;
    }

    unsigned shown_generic() const {
      return shown_generic_;
    }

    unsigned unrecorded_presents() const {
      return unrecorded_;
    }

    unsigned stalls() const {
      return stalls_;
    }

  private:
    // The first legal Present without depth while disarmed, for frames whose
    // sources meet no limitation and for frames whose sources all meet one.
    std::uint64_t &since(const supply &frame) {
      return frame.limit == limitation::none ? since_[0] : since_[1];
    }

    // The freshness rule's supply history: a real Present's frame's copies
    // the presenting queue can read, its own after that frame and those of
    // another queue once the judge saw them complete (done, at a look after
    // an earlier Present). A copy older than the source age (and the slack)
    // can be newer than no copy a Present may show.
    struct readable_copy {
      std::uint64_t stamp {}, frame {};
      std::uint8_t sources {};
      std::function<bool()> completed;
      bool done {};
      // supply::completion::relied, and whether a late signal of it made the
      // copy unholdable (late_signal).
      const void *relied {};
      bool conflicted {};
      // Recorded at its call on the presenting queue (supply::immediate):
      // readable from its own frame's Present.
      bool immediate {};
      // Each source's capture id with copy_fg_input (supply::capture), by
      // source bit; 0 unknown.
      std::array<std::uint64_t, 4> capture {};
      // An FG hold kept it off-screen (supply::fg_held): readable only at a
      // Present after FG Off (fg_off_).
      bool fg_held {};
    };

    // Whether copy is certainly readable at this Present: complete at an
    // earlier look (another queue's), or its frame's own Present passed
    // (or was this one, for a copy recorded at its call); never a conflicted
    // one, nor one an FG hold keeps off-screen while FG is on.
    bool readable_now(const readable_copy &copy) const {
      return !copy.conflicted && (!copy.fg_held || fg_off_) &&
        (copy.completed ? copy.done : copy.frame < frames_ || copy.immediate);
    }

    void readable(const supply &frame, std::uint64_t now) {
      ++frames_;
      // Each copy by when it was made (supply::made).
      const auto stamp = [&](std::uint8_t sources) {
        const auto value = frame.made_tick(sources);
        return value ? value : now;
      };
      // The capture ids of these sources' copies.
      const auto captures = [&](std::uint8_t sources) {
        std::array<std::uint64_t, 4> out {};
        for (unsigned i = 0; i != out.size(); ++i) {
          if (sources & (1u << i)) {
            out[i] = frame.capture[i];
          }
        }
        return out;
      };
      const auto local = std::uint8_t(frame.legal & api_sources & ~frame.limited & ~frame.elsewhere);
      // The frame's own copies, and those an FG hold keeps off-screen
      // (supply::fg_held), readable only after FG Off.
      for (const bool held : {false, true}) {
        if (const auto own = std::uint8_t(local & (held ? frame.fg_held & frame.owned : ~frame.owned))) {
          if (const auto at_call = std::uint8_t(own & frame.immediate)) {
            copies_.push_back({stamp(at_call), frames_, at_call, {}, false, nullptr, false, true, captures(at_call), held});
          }
          if (const auto later = std::uint8_t(own & ~frame.immediate)) {
            copies_.push_back({stamp(later), frames_, later, {}, false, nullptr, false, false, captures(later), held});
          }
        }
      }
      for (const auto &value : frame.completions) {
        std::array<std::uint64_t, 4> ids {};
        for (unsigned i = 0; i != ids.size(); ++i) {
          if (value.sources & (1u << i)) {
            ids[i] = value.capture;
          }
        }
        copies_.push_back({stamp(value.sources), frames_, value.sources, value.probe, value.done_before_present, value.relied, false, false, ids, value.fg_held});
      }
      while (!copies_.empty() && copies_.front().stamp + sunshine_freshness::source_age_ms + tick_slack_ms < now) {
        copies_.pop_front();
      }
    }

    // Each copy from another queue: whether its work completed by now.
    void look() {
      for (auto &copy : copies_) {
        copy.done = copy.done || (copy.completed && copy.completed());
      }
    }

    // Whether the copy the last real Present showed (its rung's newest copy
    // stamped at or before the record's depth tick) was read through a fence
    // a late signal reached since: O1 stops holding exactly that copy.
    bool shown_conflicted() const {
      const readable_copy *shown {};
      for (const auto &copy : copies_) {
        if ((copy.sources & shown_copy_.sources) && copy.stamp <= shown_copy_.tick && (!shown || copy.stamp >= shown->stamp)) {
          shown = &copy;
        }
      }
      return shown_copy_.tick && shown && shown->conflicted;
    }

    // Whether this Present could hold a copy: one certainly readable at it
    // (readable_after, never a conflicted one) younger than the source age.
    // Within the slack of its expiry it counts as expired: the add-on's
    // capture tick is up to the slack after the fixture's stamp.
    bool holdable(std::uint64_t now) const {
      const auto newest = readable_after(0, api_sources).second;
      return newest && now + tick_slack_ms < newest + sunshine_freshness::source_age_ms;
    }

    // The newest copy of sources certainly readable at this Present (as
    // readable_after counts them) of the same kind as the shown copy
    // (copy_fg_input) and with a capture id above it, with its kind flag;
    // 0 when there is none, or when the shown copy is not one the judge knows
    // (its kind unknown).
    // lineage: of either kind (an FG copy held after FG Off), whether or not
    // the judge knows the shown copy.
    std::uint64_t readable_newer(std::uint64_t shown, std::uint8_t sources, bool lineage = false) const {
      if (!shown) {
        return 0;
      }
      std::uint64_t kind = UINT64_MAX;
      for (const auto &copy : copies_) {
        for (const auto value : copy.capture) {
          if (value && (value & ~copy_fg_input) == shown) {
            kind = value & copy_fg_input;
          }
        }
      }
      if (kind == UINT64_MAX && !lineage) {
        return 0;
      }
      std::uint64_t newest {};
      for (const auto &copy : copies_) {
        if (!readable_now(copy)) {
          continue;
        }
        for (unsigned i = 0; i != copy.capture.size(); ++i) {
          const auto value = copy.capture[i];
          if ((sources & copy.sources & (1u << i)) && value && (lineage || (value & copy_fg_input) == kind) &&
              (value & ~copy_fg_input) > shown && (value & ~copy_fg_input) > (newest & ~copy_fg_input)) {
            newest = value;
          }
        }
      }
      return newest;
    }

    // The frames that supplied a copy of sources certainly readable at this
    // Present after tick, and the newest one's stamp.
    std::pair<unsigned, std::uint64_t> readable_after(std::uint64_t tick, std::uint8_t sources) const {
      unsigned frames {};
      std::uint64_t last {}, newest {};
      for (const auto &copy : copies_) {
        if (!(copy.sources & sources) || copy.stamp <= tick || !readable_now(copy)) {
          continue;
        }
        frames += copy.frame != last;
        last = copy.frame;
        newest = std::max(newest, copy.stamp);
      }
      return {frames, newest};
    }

    // armed_: a Present showed depth since the last loss; showing_: the
    // previous Present did; fresh_: the last real Present showed a fresh
    // copy of a frame whose legal sources the presenting queue reads;
    // warming_: a warm-up (the session's start, a stall or a new source)
    // gives legal sources establish_ms.
    bool armed_ {}, showing_ {}, fresh_ {}, warming_ {true};
    // The Present being judged comes after FG Off (supply::fg_off).
    bool fg_off_ {};
    std::array<std::uint64_t, 2> since_ {};
    // The supply of the frame the last real Present presented, and the last
    // three real Presents' ticks, newest first.
    supply presented_;
    // What the last real Present showed from an API rung: its record's depth
    // tick and that rung's sources (none: it showed no API copy).
    struct shown_copy {
      std::uint64_t tick {};
      std::uint8_t sources {};
    } shown_copy_;
    std::array<std::uint64_t, 3> real_presents_ {};
    // The copy the last real Present that showed API depth showed, and its
    // API (the handover rule).
    struct shown_api {
      std::uint64_t capture {};
      sunshine_game3d::source_api api {sunshine_game3d::source_api::none};
    } handover_;
    std::uint64_t last_legal_ {}, last_present_ {}, newest_api_ {}, newest_generic_ {}, newest_readable_ {}, newest_limited_ {};
    unsigned presents_ {}, legal_presents_ {}, shown_ {}, shown_generic_ {}, unrecorded_ {}, stalls_ {};
    std::uint64_t frames_ {};
    std::deque<readable_copy> copies_;
    std::array<row_tally, gap_count> rows_ {};
    std::vector<violation> pending_;
  };
}  // namespace game3d_continuity
