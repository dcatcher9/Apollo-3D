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
// within that age is no row's; a row whose limitation is such a drop (C8) is
// the exception (drops_hold). A violation no row records fails
// its run, and so does a row whose
// limitation was supplied in a run without a violation (the gap closed: the
// fix removes the row).
//
// Two rows are the limitations the owner approved until Step 4 (C1, C2) and
// one carries a matrix row's recorded gap over frames (C5). The others are
// defects these tests found (found: when and by which test), excused only
// until the owner decides each (README, Found defects): a strict run
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

  // A known limitation the frame's legal source meets; a gap row names it.
  enum class limitation : std::uint8_t {
    none,
    // API depth recorded on a queue other than the presenting one: an SR
    // evaluation's on the game's queue, or DLSS-G's own on its compute queue.
    cross_queue_sr,
    // A Generic depth-stencil view drawn on a queue other than the
    // presenting one, whose only boundary is the end of the frame.
    cross_queue_generic,
    // An OnlyValidNow tag outside FG scope copied at its call, which the same
    // frame's evaluation then refuses for its lifetime.
    tag_call_then_refused_evaluate,
    // An FG tag's copy deferred to its frame's Present in a frame DLSS-G does
    // not evaluate, within the source age of its last evaluation.
    deferred_copy_awaits_dlssg,
    // FG turned off within the source age of the FG copy the display holds,
    // with only SR depth on another queue (cross_queue_sr) to follow it.
    fg_off_drops_fg_copy,
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
    limitation limit {limitation::none};
    std::uint64_t tick {};

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
  };

  constexpr const char *name(violation_kind value) {
    switch (value) {
      case violation_kind::gap:
        return "gap";
      case violation_kind::unestablished:
        return "unestablished";
      case violation_kind::dropped_hold:
        return "dropped_hold";
      default:
        return "stale";
    }
  }

  constexpr const char *name(limitation value) {
    switch (value) {
      case limitation::cross_queue_sr:
        return "cross_queue_sr";
      case limitation::cross_queue_generic:
        return "cross_queue_generic";
      case limitation::tag_call_then_refused_evaluate:
        return "tag_call_then_refused_evaluate";
      case limitation::deferred_copy_awaits_dlssg:
        return "deferred_copy_awaits_dlssg";
      case limitation::fg_off_drops_fg_copy:
        return "fg_off_drops_fg_copy";
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
  };

  // Known limitations: a violation whose frame's legal source meets a row's
  // limitation is that row's (gap and unestablished; staleness has none).
  struct gap_row {
    const char *id;
    limitation limit;
    // The limitation itself drops a copy the presenting queue could read
    // within the source age (C8): the row records such a loss too. Any
    // other row records a loss only once no such copy is that young.
    bool drops_hold;
    // A defect these tests found, awaiting the owner's decision: the date
    // and the test that found it. nullptr: an approved limitation, or a
    // recorded matrix gap over frames.
    const char *found;
    const char *owner_note;
  };

  inline constexpr gap_row known_gaps[] {
    {"C1-cross-queue-sr", limitation::cross_queue_sr, false, nullptr, "Until Step 4's fence observation. An SR snapshot copied on a queue other than the presenting one is readable only once its recording retired and its producer fence completed (streamline_depth_capture.cpp completed_snapshot, submission_status), never through the game's own wait for that queue: in Assassin's Creed Shadows 110-235 ms after its evaluation, about 30 no_completed_snapshot losses a session, mostly where a stray FG copy hands back to SR (d8c52e3b). The fixture's game retires its recordings later than the source age, so every Present loses that depth once the presenting queue's last copy ages out. DLSS-G's own copies on its COMPUTE list (N7) are read the same way, once DLSS-G reset that recording (accept_compute_copy, three of its evaluations later with its ring of three), and the session timelines tag their frames with this row too, the viewport's SR input among them: a frame whose only FG input is such a copy supplies its live FG scope, which keeps that SR input out while the copy is in flight (the second kind of frame of C4, closed 10-26). The row records a loss only once no copy the presenting queue could read (a frame's legal source that meets no limitation) is younger than the source age: a loss within it, such as a held FG copy dropped for an SR snapshot still in flight on the other queue (fixed by 88b45edf), is a violation. Observing the game's fence signals and waits orders the copy before the Present; that fix shows depth here and removes this row."},
    {"C2-cross-queue-generic", limitation::cross_queue_generic, false, nullptr, "Until Step 4's fence hooks. Generic's end-of-frame copy runs on the presenting queue, and a proof belongs to the queue whose executed list made it, so a depth-stencil view drawn on another queue with no boundary but the end of the frame is refused (missing_state; 18e5d922, docs/reshade-sbs.md, the last proven state). Seven of the eight games of the 10-06 to 10-23 logs render on another queue. Either owner option (accept the other queue's proof once fence hooks show the presenting queue ordered after it, or copy on the proving queue) shows depth here and removes this row."},
    {"C5-S1-evaluate", limitation::tag_call_then_refused_evaluate, false, nullptr, "The API contract matrix's S1-evaluate row (test_game3d_api_contract_known_gaps.h) over frames: an OnlyValidNow tag outside FG scope copied at its call (S1), whose same-frame evaluation refuses the tag (S3), loses that copy, so a run of such frames is mono once the earlier depth ages out. That row's fix (keep a tag-call copy when a later point of its frame refuses the same tag for its lifetime) removes this one."},
    {"C7-deferred-copy-awaits-dlssg", limitation::deferred_copy_awaits_dlssg, false, "2026-10-25, T2 (the session timelines)", "Found by the session timelines (T2), outside the planned limitations; owner decision. Under frame generation an FG tag's copy deferred to its frame's Present (R2: no command list, or a write state its list records) is made at the first Present after DLSS-G evaluated a frame following the tag, as long as DLSS-G evaluated anything within the source age (streamline_camera_probe.cpp present_deferred: evaluated, reached), refused evaluations included. When DLSS-G stops evaluating while the game keeps tagging (generation paused, its evaluations dropped, or DLSS-G evaluating some frames only), each deferred copy waits until the source age after DLSS-G's last evaluation, while the copy the display holds ages out first: its Presents are mono although every frame's tag was copyable at its Present (witness '20*sr=local; fg=on,tag=present_bare; 3*dlssg=compute_unobserved; 40*tag=present_bare'; the soak's seeds 3962e7a36baf19cf and e1bfe1504e0dd344). Waiting for DLSS-G only while it evaluates every frame (an evaluation since the previous tagged frame's Present), or no longer than the held copy can stand in, removes this row."},
    {"C8-fg-off-drops-fg-copy", limitation::fg_off_drops_fg_copy, true, "2026-10-25, T4 (the session timelines)", "Found by the session timelines (T2) once C1 recorded only the losses its limitation explains; owner decision. When FG is turned off while the display holds an FG copy younger than the source age and the viewport's SR depth is copied on another queue (C1), the provider drops the FG copy as a changed source (decide_display: source_changed, the SR snapshot pending) instead of holding it, so the Presents until that SR copy is readable are mono although a copy the presenting queue could show was within the source age (witness '20*sr=local; fg=on,tag=present_list; fg=off,sr=other_queue; 10*sr=other_queue'; reproducer 'fg=on; tag=present_bare; fg=off,sr=other_queue'). 88b45edf's hold (frame_generation_holds) keeps an FG copy over a pending SR copy only while FG is on, and its commit left FG Off unchanged. In Assassin's Creed Shadows, whose SR copies run on another queue, each FG Off would lose depth until SR is readable (110-235 ms). Holding the FG copy across FG Off within the source age (as any source's missing frame is held), or Step 4's fence observation (C1), removes this row."},
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
    if (value.kind == violation_kind::stale || value.frame.limit == limitation::none) {
      return -1;
    }
    for (std::size_t i = 0; i != gap_count; ++i) {
      if (known_gaps[i].limit != value.frame.limit) {
        continue;
      }
      // A limitation loses depth once the presenting queue's newest copy
      // aged out (C1's note); a loss while that copy could still be held is
      // no row's but a drop's own (drops_hold).
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
    char text[600];
    std::snprintf(text, sizeof(text), "%s in %s at present=%llu tick=%llu: frame legal=0x%x permitted=0x%x limitation=%s supplied_at=%llu age_ms=%llu readable_age_ms=%lld; record owner=%s state=%s reason=%s", name(value.kind), value.where, static_cast<unsigned long long>(value.present), static_cast<unsigned long long>(value.tick), unsigned(value.frame.legal), unsigned(value.frame.permitted), name(value.frame.limit), static_cast<unsigned long long>(value.frame.tick), static_cast<unsigned long long>(value.age_ms), value.readable_age_ms == UINT64_MAX ? -1ll : static_cast<long long>(value.readable_age_ms), sunshine_game3d::name(value.owner), sunshine_game3d::name(value.state), value.reason ? value.reason : "");
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
  // each, and what a strict run does with them.
  inline std::string found_defects(const std::array<row_tally, gap_count> &rows) {
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
    std::array<row_tally, gap_count> rows_ {};
    std::vector<violation> pending_;
  };
}  // namespace game3d_continuity
