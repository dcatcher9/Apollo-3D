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
// frame had no legal source may hold within the source age, then go mono.
//
// A judge reads two things per Present: the frame record the test add-on
// exports (SunshineGame3DTestFrameRecord, docs/reshade-sbs.md, Frame
// record) and the fixture's own supply log, what the frame it presents
// supplied as the fixture knows it (never what the add-on reports). A
// generated Present, which the game makes while it records a later frame,
// presents no frame of its own and supplies nothing.
//
// Warm-up: the judge is armed by a Present that shows depth. A Present that
// shows none disarms it when its frame had no legal source (the mono the
// invariant permits) or when it is a violation (one loss, one violation).
// While disarmed, legal sources have establish_ms to show depth, counted
// from the first one after the last Present that showed depth, or after a
// source-age gap in legal sources (a new source). A Present at least the
// source age after the previous one also disarms it first: every hold
// expired with no Present in between (a stall of the game or the system),
// and a pipelined source (DLSS-G's ring) has nothing readable within the
// age until it refills.
//
// A violation of a known limitation is a known gap: the row that names the
// limitation records it, and the fixture tags the frames that meet it. A
// violation no row records fails its run, and so does a row whose
// limitation was supplied in a run without a violation (the gap closed: the
// fix removes the row). A random session timeline may supply a limitation
// without losing depth, so the rows the session timelines found
// (test_game3d_session_timeline_runtime.cpp) are kept by their witness
// timelines instead, each of which must lose depth on every run. The rows
// follow the API contract matrix's known-gap discipline
// (test_game3d_api_contract_known_gaps.h) with their own key, the
// limitation, since no oracle cell describes a session.
#include "freshness.h"
#include "game3d_frame_record.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
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
    // The game created a command queue since the previous Present.
    queue_created,
    // SR depth kept out by an FG scope that is still live after the copy it
    // can show aged out.
    fg_scope_outlives_copy,
    // An OnlyValidNow tag outside FG scope copied at its call, which the same
    // frame's evaluation then refuses for its lifetime.
    tag_call_then_refused_evaluate,
    // DLSS-G's evaluation in a frame without an FG tag, within the source
    // age of an earlier frame's FG tag copy.
    dlssg_defers_to_tag_copy,
    // An FG tag's copy deferred to its frame's Present in a frame DLSS-G does
    // not evaluate, within the source age of its last evaluation.
    deferred_copy_awaits_dlssg,
  };

  // What one frame supplied, stamped when the fixture made the first call
  // that supplies it (no later than the add-on's own capture tick).
  struct supply {
    // Sources the contract requires the add-on to be able to show.
    std::uint8_t legal {};
    // Sources it may show but need not: an EITHER outcome, or a Generic view
    // the add-on leaves aside while an API source owns the queue (H).
    std::uint8_t permitted {};
    limitation limit {limitation::none};
    std::uint64_t tick {};

    void add(std::uint8_t sources, std::uint64_t now, limitation meets = limitation::none) {
      legal |= sources;
      if (meets != limitation::none) {
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

  enum class violation_kind : std::uint8_t {
    // An armed judge: the frame had a legal source and the Present showed
    // no depth.
    gap,
    // Legal sources established no depth within establish_ms.
    unestablished,
    // A rung showed pixels at least the source age (and the slack) after
    // the newest legal or permitted source of its kind (API or Generic).
    stale,
  };

  constexpr const char *name(violation_kind value) {
    switch (value) {
      case violation_kind::gap:
        return "gap";
      case violation_kind::unestablished:
        return "unestablished";
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
      case limitation::queue_created:
        return "queue_created";
      case limitation::fg_scope_outlives_copy:
        return "fg_scope_outlives_copy";
      case limitation::tag_call_then_refused_evaluate:
        return "tag_call_then_refused_evaluate";
      case limitation::dlssg_defers_to_tag_copy:
        return "dlssg_defers_to_tag_copy";
      case limitation::deferred_copy_awaits_dlssg:
        return "deferred_copy_awaits_dlssg";
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
  };

  // Known limitations: a violation whose frame's legal source meets a row's
  // limitation is that row's (gap and unestablished; staleness has none).
  struct gap_row {
    const char *id;
    limitation limit;
    const char *owner_note;
  };

  inline constexpr gap_row known_gaps[] {
    {"C1-cross-queue-sr", limitation::cross_queue_sr, "Until Step 4's fence observation. An SR snapshot copied on a queue other than the presenting one is readable only once its recording retired and its producer fence completed (streamline_depth_capture.cpp completed_snapshot, submission_status), never through the game's own wait for that queue: in Assassin's Creed Shadows 110-235 ms after its evaluation, about 30 no_completed_snapshot losses a session, mostly where a stray FG copy hands back to SR (d8c52e3b). The fixture's game retires its recordings later than the source age, so every Present loses that depth once the presenting queue's last copy ages out. DLSS-G's own copies on its COMPUTE list (N7) are read the same way, once DLSS-G reset that recording (accept_compute_copy, three of its evaluations later with its ring of three), and the session timelines tag their frames with this row too. Observing the game's fence signals and waits orders the copy before the Present; that fix shows depth here and removes this row."},
    {"C2-cross-queue-generic", limitation::cross_queue_generic, "Until Step 4's fence hooks. Generic's end-of-frame copy runs on the presenting queue, and a proof belongs to the queue whose executed list made it, so a depth-stencil view drawn on another queue with no boundary but the end of the frame is refused (missing_state; 18e5d922, docs/reshade-sbs.md, the last proven state). Seven of the eight games of the 10-06 to 10-23 logs render on another queue. Either owner option (accept the other queue's proof once fence hooks show the presenting queue ordered after it, or copy on the proving queue) shows depth here and removes this row."},
    {"C3-queue-created", limitation::queue_created, "Found by this invariant (T1), outside the planned limitations; owner decision. A command queue created mid-session gives the native observer a queue entry point to hook; installing it (streamline_depth_capture.cpp poll, native_observer::install_pending) invalidates every capture as an observer loss, so the next Present's fresh copy finds its consumer recording invalid (consumer=recording_invalid loss=global_observation_loss), its display copy fails, and a failed fresh copy invalidates the display cache instead of holding (streamline_depth_provider.cpp: display_copy_failed). That Present is mono although a copy within the source age (16-47 ms old in the fixture) is retained in private display storage already ordered on the presenting queue. A game hits it whenever it creates a queue while depth is live, as a DLSS-G swapchain does the first time frame generation presents. Holding the retained copy across an observer loss, or installing hooks without invalidating already finished copies, removes this row."},
    {"C4-fg-scope-outlives-copy", limitation::fg_scope_outlives_copy, "Found by the session timelines (T2), outside the planned limitations; owner decision. An FG scope stays live for fg_silent_ms after its newest accepted copy (streamline_depth_capture.cpp frame_generation_scope_live) unless a later frame's FG call marks a miss (frame_generation_missed), and while it is live the same viewport's SR evaluations are not nominated (streamline_camera_probe.cpp source_path_selected); the provider holds that copy only for the source age. Three kinds of frame mark no miss, so after the copy's hold expires each Present is mono until fg_silent_ms although every frame's SR input was copyable: a frame that makes no FG call at all (no FG depth tag and no DLSS-G evaluation: generation paused, or a game that tags FG depth only on frames it generates; witness '20*sr=local; fg=on,dlssg=direct; 40*sr=local', mono from about 220 ms to 1000 ms after the copy); a frame whose only FG input is DLSS-G's compute copy, accepted once readable (C1) with its own older tick; and a frame whose FG tag is refused or whose deferred copy an untag cancels within fg_silent_ms of DLSS-G's own last copy, which miss_frame_generation leaves to that evaluation (evaluation_supply) although no evaluation comes (the soak's seeds f0857357b2b8f22d, f03cf607addde182, 8eee8807bd66e868 and 40e5fd9a22d273bc). The oracle's SR input is a CAPTURE whatever the FG scope (S5, S20). Either removes this row: count a frame whose SR evaluation of the viewport comes without a copied FG input as missed, or let the scope's liveness end with its newest copy's source age."},
    {"C5-S1-evaluate", limitation::tag_call_then_refused_evaluate, "The API contract matrix's S1-evaluate row (test_game3d_api_contract_known_gaps.h) over frames: an OnlyValidNow tag outside FG scope copied at its call (S1), whose same-frame evaluation refuses the tag (S3), loses that copy, so a run of such frames is mono once the earlier depth ages out. That row's fix (keep a tag-call copy when a later point of its frame refuses the same tag for its lifetime) removes this one."},
    {"C6-dlssg-defers-to-tag-copy", limitation::dlssg_defers_to_tag_copy, "Found by the session timelines (T2), outside the planned limitations; owner decision. DLSS-G's own evaluation adds no copy while an FG tag copy of its scope is younger than the source age (streamline_camera_probe.cpp tag_copy_recent: tag_copied), which the precedence rule meant for the same frame's tag (an FG tag copies its frame first) but which also covers later frames without a tag. Such an evaluation is neither a copy nor a miss (frame_generation_missed), so for the source age DLSS-G's fresher depth is not copied (the earlier tag copy is held), and the scope falls silent only at the first evaluation after it, when that copy's hold has just expired: the frame whose SR evaluation came before its DLSS-G evaluation (the order a game evaluates them in) has no SR nomination, and its Present is mono (witness '20*sr=local; fg=on,tag=present_list; 30*sr=local,dlssg=compute_unobserved'; the soak's seeds 059c8f969c73642a, b7d533b4c7fd52d7 and f535aa6670376ad9). Keying the precedence to the evaluated frame (its token or call sequence) instead of the source age removes this row."},
    {"C7-deferred-copy-awaits-dlssg", limitation::deferred_copy_awaits_dlssg, "Found by the session timelines (T2), outside the planned limitations; owner decision. Under frame generation an FG tag's copy deferred to its frame's Present (R2: no command list, or a write state its list records) is made at the first Present after DLSS-G evaluated a frame following the tag, as long as DLSS-G evaluated anything within the source age (streamline_camera_probe.cpp present_deferred: evaluated, reached), refused evaluations included. When DLSS-G stops evaluating while the game keeps tagging (generation paused, its evaluations dropped, or DLSS-G evaluating some frames only), each deferred copy waits until the source age after DLSS-G's last evaluation, while the copy the display holds ages out first: its Presents are mono although every frame's tag was copyable at its Present (witness '20*sr=local; fg=on,tag=present_bare; 3*dlssg=compute_unobserved; 40*tag=present_bare'; the soak's seeds 3962e7a36baf19cf and e1bfe1504e0dd344). Waiting for DLSS-G only while it evaluates every frame (an evaluation since the previous tagged frame's Present), or no longer than the held copy can stand in, removes this row."},
  };
  inline constexpr std::size_t gap_count = std::size(known_gaps);

  constexpr int known(const violation &value) {
    if (value.kind == violation_kind::stale || value.frame.limit == limitation::none) {
      return -1;
    }
    for (std::size_t i = 0; i != gap_count; ++i) {
      if (known_gaps[i].limit == value.frame.limit) {
        return int(i);
      }
    }
    return -1;
  }

  inline std::string describe(const violation &value) {
    char text[512];
    std::snprintf(text, sizeof(text), "%s in %s at present=%llu tick=%llu: frame legal=0x%x permitted=0x%x limitation=%s supplied_at=%llu age_ms=%llu; record owner=%s state=%s reason=%s", name(value.kind), value.where, static_cast<unsigned long long>(value.present), static_cast<unsigned long long>(value.tick), unsigned(value.frame.legal), unsigned(value.frame.permitted), name(value.frame.limit), static_cast<unsigned long long>(value.frame.tick), static_cast<unsigned long long>(value.age_ms), sunshine_game3d::name(value.owner), sunshine_game3d::name(value.state), value.reason ? value.reason : "");
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

  class judge {
  public:
    struct row_tally {
      unsigned exercised {}, violations {};
    };

    // One Present: its frame record and the supply of the frame it presents
    // (empty for a generated Present), in the script or phase where.
    void present(const sunshine_game3d::frame_record &record, const supply &frame, const char *where) {
      using sunshine_game3d::source_slot;
      ++presents_;
      const auto now = record.tick_ms;
      if (last_present_ && now >= last_present_ + sunshine_freshness::source_age_ms) {
        ++stalls_;
        armed_ = false;
        warm_since_ = 0;
      }
      last_present_ = now;
      const auto stamp = frame.tick ? frame.tick : now;
      const auto offered = std::uint8_t(frame.legal | frame.permitted);
      if (offered & api_sources) {
        newest_api_ = stamp;
      }
      if (offered & generic_sources) {
        newest_generic_ = stamp;
      }
      if (frame.legal) {
        ++legal_presents_;
        for (std::size_t i = 0; i != gap_count; ++i) {
          rows_[i].exercised += known_gaps[i].limit == frame.limit;
        }
      }
      const auto owner = record.depth_owner;
      const auto &row = owner == source_slot::none ? record[source_slot::hold] : record[owner];
      const auto report = [&](violation_kind kind, std::uint64_t age) {
        violation value {kind, where, record.present, now, frame, age, owner, row.state, row.reason};
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
          report(violation_kind::stale, age);
        }
        armed_ = true;
        warm_since_ = 0;
        last_legal_ = frame.legal ? now : last_legal_;
        return;
      }
      if (!frame.legal) {
        armed_ = false;
        return;
      }
      if (armed_) {
        report(violation_kind::gap, 0);
        armed_ = false;
        warm_since_ = now;
      } else if (!warm_since_ || !last_legal_ || now - last_legal_ >= sunshine_freshness::source_age_ms) {
        warm_since_ = now;
      } else if (now - warm_since_ > establish_ms) {
        report(violation_kind::unestablished, now - warm_since_);
        warm_since_ = now;
      }
      last_legal_ = now;
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
    bool armed_ {};
    std::uint64_t warm_since_ {}, last_legal_ {}, last_present_ {}, newest_api_ {}, newest_generic_ {};
    unsigned presents_ {}, legal_presents_ {}, shown_ {}, shown_generic_ {}, unrecorded_ {}, stalls_ {};
    std::array<row_tally, gap_count> rows_ {};
    std::vector<violation> pending_;
  };
}  // namespace game3d_continuity
