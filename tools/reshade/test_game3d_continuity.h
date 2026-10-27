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
// that no live FG scope keeps out) from its frame's Present, and a copy
// recorded on another queue once its work completed, if the contract copies
// it out (docs/reshade-sbs.md, Copy-out on the producer queue;
// supply::completions). The record names the shown copy by its capture tick
// (frame_record::depth_tick_ms). Showing older depth while newer copies were
// readable is stale_shown: no row excuses it, as no known limitation makes a
// readable copy unreadable.
//
// A judge reads two things per Present: the frame record the test add-on
// exports (SunshineGame3DTestFrameRecord, docs/reshade-sbs.md, Frame
// record) and the fixture's own supply log, what the frame it presents
// supplied as the fixture knows it (never what the add-on reports). A
// generated Present supplies nothing of its own.
//
// Warm-up: the judge is armed by a Present that shows depth. A Present that
// shows none disarms it when its frame had no legal source (the mono the
// invariant permits) or when it is a violation; a generated Present neither
// arms nor disarms it. While disarmed, a frame's legal sources have one
// source age to show depth, counted from the first such Present, and
// establish_ms only during a warm-up: the session's start, a stall (a
// Present at least the source age after the previous one: every hold
// expired with no Present in between, and a pipelined source such as
// DLSS-G's ring has nothing readable within the age until it refills) or a
// new source (a source-age gap in legal sources). Frames whose sources meet
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
// One row is the limitation the owner approved until Step 4 (C2). C1, API
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
    limitation limit {limitation::none};
    std::uint64_t tick {};
    // One probe per copy recorded on another queue, with its source: true
    // once its producer's work completed, when the presenting queue can read
    // its copy-out (every such copy is copied out since U-C1b, 10-27).
    std::vector<std::pair<std::uint8_t, std::function<bool()>>> completions;

    void add(std::uint8_t sources, std::uint64_t now, limitation meets = limitation::none) {
      legal |= sources;
      if (meets != limitation::none) {
        limited |= sources;
        limit = meets;
      }
      if (!tick) {
        tick = now;
      }
    }

    void allow(std::uint8_t sources, std::uint64_t now) {
      permitted |= sources;
      if (!tick) {
        tick = now;
      }
    }

    // A legal copy recorded on another queue; probe: its completion.
    void add_elsewhere(std::uint8_t sources, std::uint64_t now, std::function<bool()> probe = {}) {
      add(sources, now);
      elsewhere |= sources;
      completes(sources, std::move(probe));
    }

    void completes(std::uint8_t sources, std::function<bool()> probe) {
      if (probe) {
        completions.emplace_back(sources, std::move(probe));
      }
    }
  };

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
  // Present after it), and a copy from another queue that the judge saw
  // complete at an earlier Present (its work completed before this Present's
  // decision). Two of the fixture's frame intervals, its own pacing (the mean
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
  };

  inline constexpr gap_row known_gaps[] {
    {"C2-cross-queue-generic", limitation::cross_queue_generic, nullptr, "Until Step 4's fence hooks. Generic's end-of-frame copy runs on the presenting queue, and a proof belongs to the queue whose executed list made it, so a depth-stencil view drawn on another queue with no boundary but the end of the frame is refused (missing_state; 18e5d922, docs/reshade-sbs.md, the last proven state). Seven of the eight games of the 10-06 to 10-23 logs render on another queue. Either owner option (accept the other queue's proof once fence hooks show the presenting queue ordered after it, or copy on the proving queue) shows depth here and removes this row."},
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
      if (value.readable_age_ms < sunshine_freshness::source_age_ms) {
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
    char text[700];
    std::snprintf(text, sizeof(text), "%s in %s at present=%llu tick=%llu: frame legal=0x%x permitted=0x%x limitation=%s supplied_at=%llu age_ms=%llu readable_age_ms=%lld shown_tick=%llu newer_frames=%u bound_ms=%llu; record owner=%s state=%s reason=%s", name(value.kind), value.where, static_cast<unsigned long long>(value.present), static_cast<unsigned long long>(value.tick), unsigned(value.frame.legal), unsigned(value.frame.permitted), name(value.frame.limit), static_cast<unsigned long long>(value.frame.tick), static_cast<unsigned long long>(value.age_ms), value.readable_age_ms == UINT64_MAX ? -1ll : static_cast<long long>(value.readable_age_ms), static_cast<unsigned long long>(value.shown_tick), value.newer_frames, static_cast<unsigned long long>(value.bound_ms), sunshine_game3d::name(value.owner), sunshine_game3d::name(value.state), value.reason ? value.reason : "");
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

    // One Present: its frame record and the supply of the frame it presents
    // (empty for a generated Present, which presents the frame the previous
    // real Present presented), in the script or phase where.
    void present(const sunshine_game3d::frame_record &record, const supply &frame, const char *where, bool generated = false) {
      using sunshine_game3d::source_slot;
      ++presents_;
      const auto now = record.tick_ms;
      if (last_present_ && now >= last_present_ + sunshine_freshness::source_age_ms) {
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
        readable(frame, now);
      }
      if (frame.limited & api_sources) {
        newest_limited_ = frame.tick ? frame.tick : now;
      }
      const auto stamp = frame.tick ? frame.tick : now;
      const auto offered = std::uint8_t(frame.legal | frame.permitted);
      if (offered & api_sources) {
        newest_api_ = stamp;
      }
      if (offered & generic_sources) {
        newest_generic_ = stamp;
      }
      if ((frame.legal & api_sources & ~frame.owned) && frame.limit == limitation::none) {
        newest_readable_ = stamp;
      }
      if (frame.legal) {
        ++legal_presents_;
        for (std::size_t i = 0; i != gap_count; ++i) {
          rows_[i].exercised += known_gaps[i].limit == frame.limit;
        }
      }
      const auto owner = record.depth_owner;
      const auto &row = owner == source_slot::none ? record[source_slot::hold] : record[owner];
      // The freshness rule reads what was certainly readable before this
      // Present's decision, then looks at the copies from other queues again.
      std::pair<unsigned, std::uint64_t> fresher {};
      if ((owner == source_slot::fg_depth || owner == source_slot::scene_depth) && record.depth_tick_ms) {
        fresher = readable_after(record.depth_tick_ms, owner == source_slot::fg_depth ? fg_depth_sources : scene_depth_sources);
      }
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
        if (const auto [newer, newest] = fresher; newer && real_presents_[2]) {
          const auto bound = freshness_bound_ms((real_presents_[0] - real_presents_[2]) / 2);
          if (newest - record.depth_tick_ms > bound) {
            violation value {violation_kind::stale_shown, where, record.present, now, frame, newest - record.depth_tick_ms, owner, row.state, row.reason};
            value.shown_tick = record.depth_tick_ms;
            value.newer_frames = newer;
            value.bound_ms = bound;
            pending_.push_back(value);
          }
        }
        // The newest source of the shown rung's kind; a tick ahead of the
        // record's (another read of the clock) is age 0.
        const auto newest = generic ? newest_generic_ : newest_api_;
        std::uint64_t age = now;
        if (newest) {
          age = now > newest ? now - newest : 0;
        }
        if (!newest || age >= sunshine_freshness::source_age_ms + tick_slack_ms) {
          report(violation_kind::stale, age, frame);
        }
        armed_ = true;
        showing_ = true;
        if (!generated) {
          fresh_ = row.state == sunshine_game3d::source_state::ready && (frame.legal & api_sources) && !frame.limited;
        }
        warming_ = false;
        since_ = {};
        last_legal_ = frame.legal ? now : last_legal_;
        return;
      }
      const bool held = std::exchange(showing_, false);
      if (generated) {
        // It presents the frame the previous Present presented and holds the
        // depth shown there while that copy is younger than the source age,
        // less the slack. A judge knows that age where the last real Present
        // showed a fresh copy of a frame whose legal sources the presenting
        // queue reads: the copy is that frame's or the previous frame's (a
        // copy deferred to a later Present, R2), so it was made after the
        // real Present two before the last, unless a copy recorded on another
        // queue (a limited source) may have become readable since. It
        // neither arms nor disarms.
        const bool cross_queue = newest_limited_ && now < newest_limited_ + sunshine_freshness::source_age_ms + tick_slack_ms;
        if (held && fresh_ && !cross_queue && real_presents_[2] && now + tick_slack_ms < real_presents_[2] + sunshine_freshness::source_age_ms) {
          report(violation_kind::dropped_hold, now - real_presents_[2], presented_);
        }
        return;
      }
      if (!frame.legal) {
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
      next.last_present_ = last_present_;
      next.newest_api_ = newest_api_;
      next.newest_generic_ = newest_generic_;
      next.newest_readable_ = newest_readable_;
      next.newest_limited_ = newest_limited_;
      next.frames_ = frames_;
      next.copies_ = std::move(copies_);
      *this = std::move(next);
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
    };

    void readable(const supply &frame, std::uint64_t now) {
      ++frames_;
      const auto stamp = frame.tick ? frame.tick : now;
      if (const auto own = std::uint8_t(frame.legal & api_sources & ~frame.owned & ~frame.limited & ~frame.elsewhere)) {
        copies_.push_back({stamp, frames_, own, {}});
      }
      for (const auto &[sources, probe] : frame.completions) {
        copies_.push_back({stamp, frames_, sources, probe, false});
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

    // The frames that supplied a copy of sources certainly readable at this
    // Present after tick, and the newest one's stamp.
    std::pair<unsigned, std::uint64_t> readable_after(std::uint64_t tick, std::uint8_t sources) const {
      unsigned frames {};
      std::uint64_t last {}, newest {};
      for (const auto &copy : copies_) {
        if (!(copy.sources & sources) || copy.stamp <= tick || (copy.completed ? !copy.done : copy.frame >= frames_)) {
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
    std::array<std::uint64_t, 2> since_ {};
    // The supply of the frame the last real Present presented, and the last
    // three real Presents' ticks, newest first.
    supply presented_;
    std::array<std::uint64_t, 3> real_presents_ {};
    std::uint64_t last_legal_ {}, last_present_ {}, newest_api_ {}, newest_generic_ {}, newest_readable_ {}, newest_limited_ {};
    unsigned presents_ {}, legal_presents_ {}, shown_ {}, shown_generic_ {}, unrecorded_ {}, stalls_ {};
    std::uint64_t frames_ {};
    std::deque<readable_copy> copies_;
    std::array<row_tally, gap_count> rows_ {};
    std::vector<violation> pending_;
  };
}  // namespace game3d_continuity
