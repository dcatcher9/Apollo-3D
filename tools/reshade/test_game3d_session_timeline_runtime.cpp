// SPDX-License-Identifier: GPL-3.0-only
// Seeded randomized frame timelines, the session-level tests' second kind
// (README, Session-level tests). The contract matrix judges one decision at
// a time; Assassin's Creed Shadows' flicker came from correct decisions
// composed over frames. A timeline is 60 to 200 real frames, each a row of
// one family of frame events (frame_events below):
// - a Streamline depth tag with or without a command list in each lifecycle,
//   through slSetTagForFrame or slSetTag, from a state the contract proves or
//   not (the state choice);
// - a null untag after the tag (by the frame's token, or by none), or one at
//   the start of the frame (the frame before's tag), or none;
// - FG turned on or off, and 1 to 3 generated Presents after the real one,
//   inside the next frame's recording, or landing inside its SR evaluation
//   while that evaluation copies its own depth input (2x to 4x); FG turned
//   off after the frame's evaluations, before its Present (fg=off_late: a
//   game's options call there, or DLSS-G's Present thread presenting
//   between the call and the frame's next SR evaluation), which seeds draw
//   for half of their Offs while FG is on, from a stream of their own
//   (10-28, so every seed keeps its other frames);
// - a dynamic-resolution step of the tagged extent;
// - the viewport's SR evaluation with its own depth input on the presenting
//   queue or the game's own direct queue, or of the frame's tag;
// - DLSS-G's own evaluation on a direct or compute list, its depth in a state
//   recorded there or not;
// - the game's HUD-less and Backbuffer tags, with or without a command list,
//   or untagged in the same frame;
// - the game's fence events that order the presenting queue after the
//   frame's work on another queue (its SR on the game's queue, DLSS-G's
//   compute copy): the presenting queue waits after the other queue's
//   signal or before it (wait before signal), or the CPU waits for it; the
//   CPU or a third queue of the game's signals that fence again (a second
//   signaller), before the Present or only after it (a late signaller);
//   each fence is created anew at each FG toggle, as a game recreates its
//   frame-generation objects. These are drawn from a stream of their own,
//   so every seed keeps the frames it had before them (10-27);
// - the frame's slow intervals, modelled rather than left to the machine's
//   load (10-27): DLSS-G's compute work, and the SR work on the game's own
//   queue (10-28), held on the GPU until the frame's Present returned (a
//   GPU-bound game, its copy in flight at its own Present), and a hitch of
//   the source age and more between frames, of the
//   frame's recording before its calls (brief, or the source age and more)
//   or after them, before its Present, or after its Present, before its
//   generated Presents. Seeds draw them at a low weight from a stream of
//   their own, as the fence events (so every seed keeps its other frames;
//   until the 10-27 review no seed drew them).
// With FG on, the game's FG options declare each frame's generated count as
// numFramesToGenerate (and DLSS-G's MultiFrameCount), as the add-on reads it
// (until the 10-27 review always 1).
// Each dimension keeps the previous frame's value with a chance or is drawn
// anew (test_contract_matrix_generator.h, sample), so a timeline has the
// stretches a game has and any frame may change anything.
//
// Each timeline runs through the native provider fixture (the official
// ReShade runtime, the test add-on and a real D3D12 device, the 2.7.30
// interposer and the NGX stand-in) on a viewport of its own, then isolation
// retires it. It is judged:
// - every Present by the continuity invariant (test_game3d_continuity.h),
//   each frame supplying what the contract oracle
//   (test_game3d_api_contract_oracle.h) names at its copy decisions: legal
//   where the oracle requires a capture, permitted where it only allows one,
//   nothing for illegal input;
// - each copy decision by the oracle: a contract class its gate records
//   (SunshineDepthTestLastRefusal) is the oracle's, for a refusal the oracle
//   requires, none is recorded for a decision it requires to capture, and a
//   required refusal is recorded unless the FG scope decided the call first;
// - each untag the oracle names as cancelling its frame's deferred copy (R7)
//   cancels the deferral its tag call made (SunshineStreamlineTestDeferrals);
// - a Present, real or generated, shows a fresh copy only while a frame
//   within the source age supplied an API source (copies recorded on another
//   queue, which become readable later, aside), and a fresh FG-input copy
//   only while one supplied an FG source;
// - every provider status line names a Present's frame record
//   (require_records_match_log).
// A violation of a frame tagged with a known limitation is that gap row's
// (test_game3d_continuity.h, known_gaps); any other fails the timeline. A
// failing timeline is printed with its seed and event list; once every
// timeline ran, the first is shrunk (shrinker below) to a minimal
// reproducer, printed as an event list that --events replays. Each row the
// timelines found has a witness timeline that must lose depth on every run,
// and each Assassin's Creed Shadows fix a regression timeline that must not.
//
// The timelines play game time (test_game3d_clock.h): once setup is done the
// test add-on's game clock engages, each Present ends at a virtual frame
// boundary of 16 ms (watched rings complete, the add-on quiesces, then game
// time moves), a hitch moves game time at once after the same barrier, and
// every stamp and window of the judges reads that clock, as the add-on does.
// Watchdogs, the timelines' frame deadlines and the shrinker's budget read
// wall time. --wall-clock never engages it: the real clock, as before.
//
// Command line: [--list] [--timeline <seed> | --events <list> | --seeds N
// [--from M]] [--only <regression or witness>] [--shard i/N]
// [--shrink-runs N] [--trace] [--wall-clock] official.dll frozenShaders
// test.addon64 freshOutput interposer.dll. Without a selection it runs the fixed set, the
// first fixed_timelines seeds, the regressions and the witnesses, as the
// registered test does; --only runs the one regression or witness of that
// name with its own checks (its minimums, a witness's loss); --trace prints
// every Present's frame record.
// An event list holds frames separated by ';', each "defaults" or
// name=value pairs separated by ',', optionally repeated as "N*frame".
#define SUNSHINE_NATIVE_PROVIDER_FIXTURE_ONLY
// The fixture defines its Windows macros before any standard header.
// clang-format off
#include "test_game3d_native_provider_runtime.cpp"
// clang-format on
#include "test_contract_matrix_generator.h"
#include "test_game3d_api_contract_oracle.h"

#include <map>
#include <numeric>

namespace {
  namespace matrix = game3d_contract;
  namespace generator = game3d_contract::generator;
  namespace refusals = sunshine_depth_refusal;
  namespace continuity = game3d_continuity;

  // The fixed set the registered test runs, and each timeline's length.
  constexpr unsigned fixed_timelines = 24, min_frames = 60, max_frames = 200;
  // The fixed set's time bound, its watchdog 10 s later, below ctest's 480 s,
  // about 1.4 times a full run (about 340 s on 11-01 with 87 rows; it was
  // about twice a full run at 235 s, after the 10-27 review's timelines and
  // the seeds' hitches, and 184 to 187 s on 10-27, when the bound was 225 s
  // and ctest's 240 s, so a loaded machine's run could reach either), and
  // how long its first failure is shrunk once every timeline ran: a failing
  // run holds the GPU about a minute longer than a passing one and prints
  // its reproducer in time. --shrink-runs outside the fixed set shrinks for
  // longer. These are real-time durations (--wall-clock); on the game clock
  // the set takes about 15 s (11-02), and the bound only ends a hang.
  constexpr std::uint64_t fixed_budget_ms = 460000, shrink_ms = 60000;

  // The frame language: one row of this family per real frame.
  namespace frame_events {
    enum dimension : unsigned {
      fg,
      drs,
      tag,
      entry,
      untag,
      sr,
      dlssg,
      state,
      generated,
      at,
      ui,
      order,
      hitch,
      dlssg_gpu,
      sr_gpu,
      fg_at,
      reads,
      dimension_count
    };

    // off_late: FG turned off after the frame's evaluations, before its
    // Present: the FG copies the frame made are retired before any Present
    // shows them, and the frame's Present comes with FG off.
    enum class fg_event {
      keep,
      on,
      off,
      off_late
    };

    // A depth tag's lifecycle (OnlyValidNow, ValidUntilPresent,
    // ValidUntilEvaluate) and command list (the game's, or none).
    enum class tag_form {
      none,
      now_list,
      present_list,
      evaluate_list,
      now_bare,
      present_bare,
      evaluate_bare
    };

    enum class entry_form {
      frame,  // slSetTagForFrame with the frame's token.
      global  // slSetTag.
    };

    enum class untag_form {
      none,
      after_frame,  // slSetTagForFrame null with the frame's token, after the tag.
      after_global,  // slSetTag null after the tag: no token tells frames apart (R7).
      before_global  // slSetTag null before the frame's tag: the frame before's.
    };

    enum class sr_form {
      none,
      local,  // Its own depth input (lifecycle 1) on the presenting queue's list.
      tag,  // No input: the frame's tag.
      other_queue  // Its own input on the game's own direct queue (known gap C1 until 10-27).
    };

    enum class dlssg_form {
      none,
      direct,  // On the game's list, DLSSG.Depth in a state recorded there.
      direct_unobserved,  // On the game's list, no transition of its own.
      compute,  // On DLSS-G's compute queue's ring, its state recorded there.
      compute_unobserved
    };

    // The state choice of the frame's depth inputs (its tag, its SR input
    // and DLSS-G's): nothing recorded (UNORDERED_ACCESS throughout), a
    // transition to NON_PIXEL_SHADER_RESOURCE recorded before the tag call
    // (back at the end of the frame), nothing recorded and nothing declared
    // (UINT_MAX), or PIXEL_SHADER_RESOURCE recorded at the tag call only
    // (back right after it), which a longer-lived tag's declaration of its
    // later use does not name (S11). Each declaration names the state at the
    // use its lifecycle names (timeline_fixture::declared_state).
    enum class state_form {
      declared,
      proven,
      unprovable,
      recorded_other
    };

    enum class generated_at {
      between,  // After the frame's real Present.
      inside,  // Inside the frame's recording, before its evaluations.
      // The first lands inside the frame's SR evaluation of its own depth
      // input, between the input's read and its copy (DLSS-G presents earlier
      // frames on its own thread meanwhile: Assassin's Creed Shadows, most
      // often at a dynamic-resolution step), the rest after that evaluation.
      evaluate
    };

    enum class ui_form {
      none,
      bare,  // HUD-less and Backbuffer, ValidUntilPresent, no command list.
      listed,  // The same on the game's list.
      untagged  // bare, then the same frame's null untag of both.
    };

    // How the game orders the presenting queue after the frame's work on
    // another queue: that queue signals its fence after the work, and the
    // presenting queue waits for that value after the signal (wait), before
    // it (wait_first: legal, it blocks until the signal), or the CPU waits
    // for the work before the frame's Present (cpu_wait). cpu: the CPU waits
    // for the work, then signals the fence one value ahead; second: the
    // presenting queue waits, and a third queue of the game's waits for the
    // value, then signals one ahead (a second signaller), both before the
    // frame's Present: the presenting queue's next wait is then reached
    // before the next frame's work, which only refusing that fence (O3)
    // keeps from a read in flight. late: the presenting queue waits, and
    // after the frame's Present the CPU or the third queue (alternately)
    // signals the waited value again, a signal first submitted after a read
    // that relied on the fence (O1's residual).
    enum class order_form {
      wait,
      wait_first,
      cpu_wait,
      cpu,
      second,
      late
    };

    // A hitch of the frame's recording (hitch_ms): brief, before its calls
    // (half the source age, which every hold outlasts); before_sources, the
    // source age and more right after the frame's stamp, before its calls;
    // after_sources, the same after its calls, before its Present; between,
    // the same before the frame's first call and its stamp (after the
    // previous frame's Presents); after_present, the same after its real
    // Present, before its generated Presents after it; before_dlssg, the
    // same after the frame's other calls, right before DLSS-G's evaluation
    // of it (a stall before a game's Present call, inside which DLSS-G
    // evaluates the frame); before_present (11-02), the judge's tick slack
    // (two clock periods and more) after the fixture's bookkeeping, right
    // before the frame's real Present (a game thread preempted between its
    // last call and its Present call: a copy the add-on makes inside that
    // Present, a tag copy deferred there, R2, reads a later tick than that
    // bookkeeping did; no longer, so that bookkeeping, read before it, stays
    // within the slack the judge allows any stamp). Seeds never draw
    // before_dlssg or before_present (hitch_weights leaves them out), so
    // every seed keeps its frames; rows name them.
    enum class hitch_form {
      none,
      brief,
      before_sources,
      after_sources,
      between,
      after_present,
      before_dlssg,
      before_present
    };

    constexpr DWORD hitch_ms(hitch_form value) {
      return value == hitch_form::brief ? DWORD(sunshine_freshness::source_age_ms / 2) : value == hitch_form::none ? 0 :
        value == hitch_form::before_present ? DWORD(continuity::tick_slack_ms) : DWORD(sunshine_freshness::source_age_ms + 150);
    }

    // When DLSS-G's compute work runs on the GPU: as soon as its queue gets
    // it, or only once the frame's Present returned (after_present: its
    // queue waits on a gate the CPU opens then, as a GPU-bound game's copy
    // is still in flight at its own Present; never with a CPU wait for it).
    enum class dlssg_gpu_form {
      prompt,
      after_present
    };

    // The same for the SR evaluation's work on the game's own queue
    // (sr=other_queue): after_present holds it behind a gate the CPU opens
    // once the frame's Present returned (never with a CPU wait for that
    // work), so its copy is readable at its own Present only by the game's
    // fence order (U-C1), and otherwise from the next Present on.
    enum class sr_gpu_form {
      prompt,
      after_present
    };

    // Where the frame's FG options event (fg=on or off) reaches the add-on:
    // before its calls, or inside its real Present, after the UI read the FG
    // mode of that Present and before depth reads it (depth_read: the
    // add-on's SunshineGame3DTestOptionsAtDepthFrame, a game whose options
    // call lands on another thread during a Present). Seeds never draw it;
    // rows name it, and a timeline that does is judged per Present for the
    // FG mode the UI and depth used (fg_modes:disagree).
    enum class fg_at_form {
      frame,
      depth_read
    };

    // Whether the add-on's metadata reads fail during the Presents made
    // while the frame is current (after its calls: inside its recording, its
    // real Present and the generated Presents after it), as a read fails
    // that lost the race with the writers on every attempt (the add-on's
    // SunshineStreamlineTestFailMetadataReads). Seeds never draw it; rows
    // name it.
    enum class reads_form {
      kept,
      lost_at_present
    };

    template<class E>
    E of(const generator::row &value, dimension d) {
      return E(value[d]);
    }

    // What one frame can carry (partial rows): an SR evaluation of the
    // frame's tag needs a tag, an entry choice and a token untag a tag of
    // that entry, a placement generated Presents (inside an SR evaluation,
    // one of the viewport's own input), and a state choice an input it
    // describes.
    bool legal(const generator::row &value) {
      const auto has = [&](dimension d) {
        return value[d] >= 0;
      };
      if (has(tag) && has(sr) && of<sr_form>(value, sr) == sr_form::tag && of<tag_form>(value, tag) == tag_form::none) {
        return false;
      }
      if (has(tag) && has(entry) && of<entry_form>(value, entry) != entry_form::frame && of<tag_form>(value, tag) == tag_form::none) {
        return false;
      }
      if (has(entry) && has(untag) && of<untag_form>(value, untag) == untag_form::after_frame && of<entry_form>(value, entry) != entry_form::frame) {
        return false;
      }
      if (has(tag) && has(sr) && has(dlssg) && has(state) && of<state_form>(value, state) != state_form::declared && of<tag_form>(value, tag) == tag_form::none && of<sr_form>(value, sr) == sr_form::none && of<dlssg_form>(value, dlssg) == dlssg_form::none) {
        return false;
      }
      if (has(sr) && has(at) && of<generated_at>(value, at) == generated_at::evaluate && of<sr_form>(value, sr) != sr_form::local) {
        return false;
      }
      return !(has(generated) && has(at) && !value[generated] && of<generated_at>(value, at) != generated_at::between);
    }

    const generator::family &family() {
      static const generator::family value {
        "timeline",
        {
          {"fg", {"keep", "on", "off", "off_late"}},
          {"drs", {"keep", "change"}},
          {"tag", {"none", "now_list", "present_list", "evaluate_list", "now_bare", "present_bare", "evaluate_bare"}},
          {"entry", {"frame", "global"}},
          {"untag", {"none", "after_frame", "after_global", "before_global"}},
          {"sr", {"none", "local", "tag", "other_queue"}},
          {"dlssg", {"none", "direct", "direct_unobserved", "compute", "compute_unobserved"}},
          {"state", {"declared", "proven", "unprovable", "recorded_other"}},
          {"generated", {"0", "1", "2", "3"}},
          {"at", {"between", "inside", "evaluate"}},
          {"ui", {"none", "bare", "listed", "untagged"}},
          {"order", {"wait", "wait_first", "cpu_wait", "cpu", "second", "late"}},
          {"hitch", {"none", "brief", "before_sources", "after_sources", "between", "after_present", "before_dlssg", "before_present"}},
          {"dlssg_gpu", {"prompt", "after_present"}},
          {"sr_gpu", {"prompt", "after_present"}},
          {"fg_at", {"frame", "depth_read"}},
          {"reads", {"kept", "lost_at_present"}},
        },
        legal,
      };
      return value;
    }

    // The frame language before the fence events and the slow intervals,
    // which generate draws as it did, so that each seed keeps its frames
    // (legal reads neither).
    const generator::family &base_family() {
      static const generator::family value = [] {
        auto out = family();
        out.dimensions.resize(order);
        return out;
      }();
      return value;
    }

    // The fence events' own weights and chance in a thousand to keep the
    // previous frame's. A fence that a CPU or second signal poisoned stays
    // so until FG toggles, so those are rarer.
    constexpr unsigned order_weights[] {12, 4, 2, 1, 1, 1};
    constexpr unsigned order_keep_per_mille = 850;

    // The slow intervals' weights, drawn anew for each frame: about one frame
    // in 80 hitches (a long hitch costs the source age and more), and
    // DLSS-G's compute work is held past its frame's Present in about one
    // compute frame in eight.
    constexpr unsigned hitch_weights[] {470, 2, 1, 1, 1, 1};
    constexpr unsigned dlssg_gpu_weights[] {7, 1};
    // The SR work on the game's queue, from a stream of its own (10-28), so
    // every seed keeps the frames it had before it: held past its frame's
    // Present in about one such frame in eight.
    constexpr unsigned sr_gpu_weights[] {7, 1};

    // Whether the frame records work on another queue that the fence events
    // order: its SR on the game's queue or DLSS-G's compute copy.
    bool elsewhere(const generator::row &value) {
      const auto d = of<dlssg_form>(value, dlssg);
      return of<sr_form>(value, sr) == sr_form::other_queue || d == dlssg_form::compute || d == dlssg_form::compute_unobserved;
    }

    // Relative weights of each dimension's values, and the chance in a
    // thousand that a frame keeps the previous frame's value. FG and dynamic
    // resolution change by one-time events; FG starts off. fg=off_late has
    // weight 0 here so that each seed keeps its frames: generate turns half
    // of the drawn Offs while FG is on into late ones, from a stream of its
    // own (until 10-28 no seed drew it, which hid the stale FG copy held
    // after a late Off, the 10-28 review's P1 and J3).
    const generator::draw &drawing() {
      static const generator::draw value {
        {
          {940, 30, 30, 0},
          {95, 5},
          {3, 2, 2, 1, 1, 2, 1},
          {1, 1},
          {5, 1, 1, 1},
          {2, 4, 1, 1},
          {4, 2, 1, 1, 1},
          {4, 2, 1, 1},
          {3, 3, 1, 1},
          {3, 1, 1},
          {3, 1, 1, 1},
        },
        {0, 0, 850, 950, 850, 850, 850, 850, 800, 900, 900},
      };
      return value;
    }

    // A game generates frames and evaluates DLS-G only with FG on: those
    // events are dropped where FG is off, then a state choice left without
    // an input it describes.
    void normalize(std::vector<generator::row> &frames) {
      bool on = false;
      for (auto &value : frames) {
        const auto event = of<fg_event>(value, fg);
        on = event == fg_event::on ? true : event == fg_event::off ? false :
                                                                     on;
        if (!on) {
          value[generated] = value[at] = value[dlssg] = 0;
        }
        // A late Off: DLSS-G evaluates the frame, and no generated Present
        // follows (its Present comes with FG off).
        if (event == fg_event::off_late) {
          value[generated] = value[at] = 0;
          on = false;
        }
        if (!legal(value)) {
          value[state] = 0;
        }
        // Fence events order only work on another queue.
        if (!elsewhere(value)) {
          value[order] = 0;
        }
        // Only DLSS-G's compute work waits for the gate, which the CPU opens
        // after the Present: never while the CPU waits for that work.
        const auto d = of<dlssg_form>(value, dlssg);
        const auto o = of<order_form>(value, order);
        if ((d != dlssg_form::compute && d != dlssg_form::compute_unobserved) || o == order_form::cpu_wait || o == order_form::cpu) {
          value[dlssg_gpu] = 0;
        }
        // So does only the SR work on the game's queue, never while the CPU
        // waits for it, nor in a frame whose DLSS-G compute evaluation waits
        // for the game's list on the presenting queue, which waits for that
        // SR work.
        if (of<sr_form>(value, sr) != sr_form::other_queue || o == order_form::cpu_wait || o == order_form::cpu ||
            d == dlssg_form::compute || d == dlssg_form::compute_unobserved) {
          value[sr_gpu] = 0;
        }
      }
    }
  }  // namespace frame_events

  struct timeline_case {
    std::uint64_t seed {};  // 0: an explicit event list.
    std::vector<generator::row> frames;
    const char *witness {};  // The known gap row a witness timeline meets.
    const char *regression {};  // The fix a regression timeline keeps.
    bool lossless {};  // regression::lossless.
    unsigned late_reads {};  // regression::late_reads.
    unsigned ui_holds {};  // regression::ui_holds.
    unsigned unidentified_holds {};  // regression::unidentified_holds.
  };

  // Each Assassin's Creed Shadows fix has a regression timeline: an event
  // list shaped like the session it fixed, which the fixed set runs after
  // its seeds and judges as any timeline. Each fails with its fix reverted
  // (README, Session-level tests); random seeds meet these shapes only by
  // chance, and a change of the event language changes every seed. So does
  // each hold the session tests' review found unjudged by a mutation.
  // A row whose per-Present trace T1 compares exactly states when the GPU
  // runs the SR work it records on the game's own queue wherever a Present
  // shows that work's copy or the one before it: sr_gpu=after_present, in
  // flight at its own Present (as the rows' shapes have it), unless the CPU
  // waits for that work before the Present (order=cpu or cpu_wait, a hitch
  // after its calls, or DLSS-G's compute evaluation of the frame, which
  // waits for the presenting queue). With the default (prompt) its
  // completion races the add-on's read at that Present: the read order
  // (completed or ordered) and, where no fence orders the read (after an FG
  // toggle, a second or CPU signal), whether that Present shows the copy or
  // holds the one before change from run to run (10-31: the T1 flakes of
  // C8-fg-off-*, O1-late-signal-presenting-copy-held, fg-off-late-* and
  // hitch-after-sources, 3 to 10 distinct traces in 10 runs of each alone,
  // and of this round's fg-off-sr-successor-older-than-held and
  // fg-on-deferred-tag-hitch-after-sources-held, 4 and 5 in 5;
  // with the form stated, one, but for one run each of C8-fg-off-color-only
  // and fg-off-sr-successor-older-than-held whose selection reason at one
  // Present the clock decided: an older SR copy reached the source age
  // there, in C8-fg-off-color-only frame 19's, 13 frames (about 205-250 ms)
  // after its Present). Rows that keep the default are judged by
  // their verdicts, which accept either outcome. DLSS-G's compute copies
  // are read from the next Present on (U-C1), so their timing changes no
  // trace.
  struct regression {
    const char *fix, *events;
    // A loss a known gap records fails it too: with its fix reverted, the
    // Presents it loses meet a row's limitation (C6's: C1, its DLSS-G compute
    // copy readable three evaluations later), and with the fix it loses none.
    bool lossless {};
    // The late reads (unordered_read's R1 count) it must make at least: a
    // regression of O1's conflict pins the read the conflict reaches (until
    // 10-28 such a timeline made it only in some runs, and a run without it
    // judged nothing of its scenario, the 10-28 review's J1).
    unsigned late_reads {};
    // The UI generated-Present holds by depth's window it must judge at
    // least (ui_frame: a generated Present that presents the frame of the
    // last UI tag's Present, offers nothing and holds its decision,
    // generated_present), so a row written for that judgement cannot pass
    // without exercising it.
    unsigned ui_holds {};
    // The holds of the earlier frame's UI decision at Presents without frame
    // identity it must make at least (ui_unidentified_now, left unjudged),
    // so a row written for that limit cannot pass where the hold no longer
    // happens (the second fix review of 11-01).
    unsigned unidentified_holds {};
  };

  constexpr regression regressions[] {
    // FG turned on over SR depth without an FG copy: the scope owns nothing
    // until FG supplies, so SR keeps the viewport (9146a468).
    {"9146a468-fg-on", "20*sr=local; fg=on,sr=local; 20*sr=local"},
    // A Present of an earlier frame inside each SR evaluation's copy of its
    // own depth input leaves that copy current (64d6d06c).
    {"64d6d06c-present-in-sr-copy", "20*sr=local; fg=on,sr=local; 20*sr=local,generated=1,at=evaluate"},
    // DLSS-G's own depth copied on its compute list (46a4510e), which takes
    // the viewport from SR only once the Present queue can read it
    // (ad66b1c8).
    {"46a4510e-ad66b1c8-dlssg-compute", "20*sr=local; fg=on,sr=local,dlssg=compute; 20*sr=local,dlssg=compute"},
    // A stray FG copy amid frames whose deferred FG copy an untag cancels,
    // with SR on the game's own queue: the FG copy is held over the SR copy
    // still in flight (88b45edf; known gap C1 once that hold ages out).
    {"88b45edf-stray-fg-copy", "20*sr=local; fg=on,tag=present_list; 15*tag=present_bare,entry=global,untag=after_global,sr=other_queue; tag=present_list; 15*tag=present_bare,entry=global,untag=after_global,sr=other_queue"},
    // Two generated Presents after each real one hold the real frame's copy
    // (decide_display's hold at a repeated frame): with that hold dropped
    // every generated Present was mono, which no seed's judge saw (review
    // mutation M1, a dropped_hold here).
    {"repeated-frame-hold", "20*sr=local; fg=on,sr=local; 30*sr=local,generated=2"},
    // A frame with no FG copy misses its FG scope as the viewport's next
    // frame begins, so the viewport's SR evaluations copy its depth (known
    // gap C4 until then). After one DLSS-G copy: frames that make no FG call,
    // and frames whose deferred FG tag copy an untag cancels, whose miss was
    // left to a DLSS-G evaluation that never came.
    {"C4-no-fg-call", "20*sr=local; fg=on,dlssg=direct; 40*sr=local"},
    {"C4-cancelled-fg-tags", "20*sr=local; fg=on,sr=local,dlssg=direct; 40*tag=present_bare,entry=global,untag=after_global,sr=local"},
    // DLSS-G's evaluation stands aside only for its own frame's FG tag copy
    // (known gap C6 until then: for the source age after any FG tag copy, so
    // its compute copy came only once that copy's hold expired, readable
    // three evaluations later): one FG tag copy, then frames whose DLSS-G
    // evaluation copies on its compute list. Lossless: with the source-age
    // rule its three lost Presents meet C1's limitation.
    {"C6-dlssg-after-tag-copy", "20*sr=local; fg=on,tag=present_list; 30*dlssg=compute", true},
    // A deferred FG tag copy waits for DLSS-G's evaluation of its frame only
    // while DLSS-G evaluated the previous deferred frame (known gap C7 until
    // then: within the source age of any evaluation, so each copy lapsed
    // unmade until that age passed, and the held copy aged out first): a
    // command-less FG tag's copy, DLSS-G evaluations (refused) without a tag,
    // then command-less FG tags without DLSS-G.
    {"C7-dlssg-stops", "20*sr=local; fg=on,tag=present_bare; 3*dlssg=compute_unobserved; 40*tag=present_bare"},
    // An FG copy is held while the viewport's SR copy that follows it is in
    // flight (known gap C8 until then: at FG Off the provider dropped it as
    // a changed source): one FG tag copy, then FG off with the viewport's SR
    // input on the game's own queue, never readable within the source age
    // here (known gap C1 once the hold ages out), and FG on again within
    // the hold, whose silent scope selects as without FG.
    {"C8-fg-off-sr-other-queue", "20*sr=local; fg=on,tag=present_list; fg=off,sr=other_queue,sr_gpu=after_present; 4*sr=other_queue,sr_gpu=after_present; fg=on,sr=other_queue,sr_gpu=after_present; 15*sr=other_queue,sr_gpu=after_present"},
    // FG Off releases the queue's API association, yet the FG copy is held
    // within the source age (known gap C8 until then: color-only Presents
    // after FG Off dropped it and Generic's decoy view took the queue, so
    // the SR copy on the game's own queue that followed within that age
    // found nothing to hold): one FG tag copy, FG off with frames that
    // supply nothing, then the viewport's SR input on the game's queue.
    {"C8-fg-off-color-only", "20*sr=local; fg=on,tag=present_list; fg=off; 10*defaults; 15*sr=other_queue,sr_gpu=after_present"},
    // An OnlyValidNow tag outside FG scope copied at its call (S1), whose
    // frame's SR evaluation of the tag refuses it for its lifetime (S3),
    // keeps that copy (known gap C5 until then, the API contract matrix's
    // S1-evaluate row: the evaluation began one without a copy, so once the
    // SR depth before aged out every Present was mono). C5's witness, through
    // slSetTagForFrame, and the same through slSetTag.
    {"C5-frame-tag-then-evaluate", "20*sr=local; 60*tag=now_list,sr=tag"},
    {"C5-global-tag-then-evaluate", "20*sr=local; 60*tag=now_list,entry=global,sr=tag"},
    // The same with the frame's untag between the tag call and the
    // evaluate, which then reads no depth tag: it keeps the tag call's copy
    // as well (until 10-26 it began an evaluation without one, so these
    // frames were mono from frame 35 as C5's were; soak seed
    // 089154be9db5f403).
    {"C5-frame-untag-then-evaluate", "20*sr=local; 30*tag=now_list,untag=after_frame,sr=tag"},
    {"C5-global-untag-then-evaluate", "20*sr=local; 30*tag=now_list,entry=global,untag=after_global,sr=tag"},
    // FG turned on with the frame's FG tag copied at its call, and a
    // generated Present before that copy is readable (inside the SR
    // evaluation, or inside the recording): the viewport's SR copy is held
    // until the scope's first readable FG copy, as one display lineage
    // (until 10-27 the live scope dropped it, FG_scope_required, at every
    // FG On; the round 3 review's soak, 13 of its 20 failures). Toggled as
    // Assassin's Creed Shadows toggles FG.
    // An FG tag of lifecycle ValidUntilEvaluate whose list records no state
    // of it is refused at its call, and the frame's SR evaluate of the tag,
    // the use its declaration names, copies it (S11c). Until 10-27 the FG
    // tag call ended that lifetime at its own return, so the evaluate read
    // the tag unsupported (missing-depth) and every such frame was mono once
    // the depth before aged out (soak seeds bf06bc26f9b88f1c,
    // 6b3b06cd02a48739), through slSetTagForFrame and slSetTag.
    {"S11c-fg-until-evaluate-tag-then-evaluate", "20*sr=local; fg=on,tag=evaluate_list,sr=tag; 20*tag=evaluate_list,sr=tag"},
    {"S11c-fg-global-until-evaluate-tag-then-evaluate", "20*sr=local; fg=on,tag=evaluate_list,entry=global,sr=tag; 20*tag=evaluate_list,entry=global,sr=tag"},
    // DLSS-G's compute copies stop while SR depth stays copyable: the first
    // frame without one is still the live scope's, and its Present can show
    // only the scope's readable copy, three DLSS-G evaluations old, which
    // expires there (known gap C1, its compute copy readable frames later;
    // the round 3 review, soak seed 76a7d0a6922d20ea). Pins that frame's
    // C1 tagging: untagged, its loss fails the set.
    {"C4-compute-copy-stops", "20*sr=local; fg=on,sr=local,dlssg=compute,generated=3; 11*sr=local,dlssg=compute,generated=3; 10*sr=local,generated=3"},
    // The game's CPU signals the presenting queue's order fence one value
    // ahead, so the next frame's wait is reached before that frame's work on
    // the game's queue: the add-on refuses that fence for good (sticky
    // poison, O3), so no Present reads that copy in flight. With the CPU
    // poison dropped from the model, unordered_read failed its frame 21
    // (shrunk to "sr=other_queue,order=cpu; sr=other_queue").
    // Lossless: every later frame's copy stays required (until 10-27 each
    // was permitted with no completion, so a mono run there passed).
    {"O3-cpu-signal-ahead", "20*sr=other_queue; sr=other_queue,order=cpu; 30*sr=other_queue", true},
    // A silent FG scope holds DLSS-G's compute copy read in flight; a CPU
    // signal of its fence conflicts that read (O1), and the frame's SR copy
    // on the presenting queue stands in at once. Until 10-27 the FG hold was
    // decided from the conflicted copy before it was dropped, so that
    // Present was mono, which R1's tag of every legal source excused.
    // The last FG frame's compute copy is held on the GPU past its Present,
    // so that Present reads it in flight through the fence the CPU signal
    // conflicts (until 10-28 the read happened only in some runs; the 10-28
    // review's J1): one late read, always.
    {"O1-silent-scope-late-conflict", "20*sr=local; fg=on,sr=local,dlssg=compute; 9*sr=local,dlssg=compute; sr=local,dlssg=compute,dlssg_gpu=after_present; sr=local; sr=local,dlssg=compute_unobserved,order=cpu; 10*sr=local", true, 1},
    // The same conflict, then SR on the game's other queue behind a fence
    // the CPU poisons: that frame's own Present cannot read its copy, so it
    // holds the SR copy shown just before (lossless), and later frames show
    // fresh SR once complete. Until the 10-27 review the fixture still kept
    // SR out for the conflicted FG copy's source age: that SR copy was not
    // holdable, so a mono Present there was excused, and stale SR passed.
    {"O1-silent-scope-late-conflict-unordered-sr", "20*sr=local; fg=on,sr=local,dlssg=compute; 9*sr=local,dlssg=compute; sr=local,dlssg=compute,dlssg_gpu=after_present; sr=local; sr=local,dlssg=compute_unobserved,order=cpu; sr=local; sr=other_queue,order=cpu; 10*sr=other_queue", true, 1},
    // An FG tag copy on the presenting queue is shown while SR is read in
    // flight from the game's other queue; a late signal of SR's fence after
    // the Present conflicts only the SR read, so the generated Present keeps
    // holding the FG copy. Until the 10-27 review the fixture tagged any
    // dropped hold there R1-late-signaller (excused at any age).
    {"O1-late-signal-presenting-copy-held", "20*sr=other_queue,sr_gpu=after_present; fg=on,tag=now_list,sr=other_queue,sr_gpu=after_present,generated=1; 5*tag=now_list,sr=other_queue,sr_gpu=after_present,generated=1; tag=now_list,sr=other_queue,sr_gpu=after_present,order=late,generated=1; 10*tag=now_list,sr=other_queue,sr_gpu=after_present,generated=1", true},
    {"fg-on-holds-sr-copy", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 5*tag=now_list,sr=local,generated=1; fg=off,sr=local; 10*sr=local; fg=on,tag=now_list,generated=1,at=inside; 5*tag=now_list,generated=1; fg=off,sr=local; 10*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 5*sr=local,generated=1"},
    // A hitch of the source age and more inside a frame, before its SR
    // evaluation (on the presenting queue's list, then on the game's own
    // queue): its Present shows that frame's own copy, made after the hitch,
    // whose source is no older than the frame (until 10-27 the judge aged it
    // by the frame's own duration, from its first call: the 10-27 full
    // check's C8-fg-off-color-only failure, continuity:stale at frame 39 with
    // age 672 ms, after a 683 ms Present gap there).
    {"hitch-before-sources", "20*sr=local; sr=local,hitch=before_sources; 10*sr=local; 5*sr=other_queue; sr=other_queue,hitch=before_sources; 10*sr=other_queue"},
    // The same hitch after the frame's SR evaluation, before its Present:
    // the frame's own copy is the depth of the frame that Present shows, so
    // it is current there however old by the clock (presented_frame). Until
    // 10-27 that Present was mono (inactive_views), and the judge excused it
    // as a stall's warm-up (it also hit the fixed set's
    // C5-global-untag-then-evaluate once under load, at frame 25).
    {"hitch-after-sources", "20*sr=local; sr=local,hitch=after_sources; 10*sr=local; 5*sr=other_queue,sr_gpu=after_present; sr=other_queue,hitch=after_sources; 10*sr=other_queue,sr_gpu=after_present"},
    // FG On in a GPU-bound game: DLSS-G's first compute copy is still in
    // flight at its own Present (dlssg_gpu=after_present), and its order
    // fence is new, so that Present cannot read it. The frame's own SR copy
    // on the presenting queue stands in (lineage_snapshot). Until 10-27 the
    // provider held only the display's retained copy there: mono once it was
    // past the source age while the frame's own SR copy was readable (seed
    // c28965dca8d7c76c's shape with SR legal: retained_depth_expired, FG
    // pending, no_completed_snapshot), and SR depth up to the source age old
    // beside the frame's own (stale_shown). Three forms: the held copy
    // expires at FG On's first Present through two brief hitches, one on a
    // frame with no source, which still holds it (a gap, no stall); a hitch
    // of the source age and more inside FG On's first frame, before its calls
    // (a gap: that frame's Present reads its own SR copy, so the hitch is no
    // warm-up); and a held copy still within the source age beside the
    // frame's own (stale_shown).
    {"fg-on-dlssg-in-flight-sr-after-hold-expired", "20*sr=local; hitch=brief; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present,hitch=brief; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    {"fg-on-dlssg-in-flight-sr-stands-in", "20*sr=local; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present,hitch=before_sources; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    {"fg-on-dlssg-in-flight-fresh-sr-over-held", "20*sr=local; hitch=brief; defaults; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    // Seed c28965dca8d7c76c's frame 38 made deterministic: after a hitch,
    // FG On's first frame supplies only DLSS-G's compute copy, in flight and
    // unordered (its SR input unprovable), and the held copy is past the
    // source age: mono is the contract there, and DLSS-G's copy shows from
    // the next Present.
    {"fg-on-only-dlssg-in-flight-mono", "20*sr=local; sr=local,hitch=after_sources; fg=on,sr=local,state=unprovable,dlssg=compute,dlssg_gpu=after_present; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    // The same frame 22 after a hitch before frame 21's calls: its SR copy,
    // made after the hitch, is about 20 ms old at P22 and held there (the
    // judge aged a copy by its frame's first call until the 10-27 review, J4,
    // so a dropped hold there passed).
    {"fg-on-only-dlssg-in-flight-held-after-hitch-before-sources", "20*sr=local; sr=local,hitch=before_sources; fg=on,sr=local,state=unprovable,dlssg=compute,dlssg_gpu=after_present; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    // FG On and a hitch of the source age and more between a frame's
    // evaluation and one of its Presents (the 10-27 review's F1 and F2,
    // open items 1 and 2): every Present of that frame, its generated ones
    // (the declared count) and its real one in either order, shows that
    // frame's own depth, held where it already showed it (frame_presenting;
    // until then the Presents after its first were mono, inactive_views or
    // retained_depth_expired, and the first Present after the pause without
    // depth restarted the stereo fade). The hitch after the evaluation with
    // the generated Present inside the next evaluation (DLSS-G's order: the
    // generated Present before the real one), after the real Present's
    // calls with the generated Presents after it (2x and 4x), after the real
    // Present before its generated one, DLSS-G's compute copy in flight at
    // FG On (the frame's SR copy stands in), and a live compute scope. And
    // the steady FG-On frame after an FG-supplying frame without an FG call
    // of its own: the live scope refuses its SR copy (owned), so after the
    // hitch nothing is readable or holdable and mono is the contract there
    // (judged so since the 10-27 review's J2 and J3, with the scope's hold
    // read at the Present's time).
    {"fg-on-hitch-generated-before-real", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 10*tag=now_list,sr=local,generated=1,at=evaluate; tag=now_list,sr=local,generated=1,at=evaluate,hitch=after_sources; 10*tag=now_list,sr=local,generated=1,at=evaluate"},
    {"fg-on-hitch-real-before-generated", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1; 10*tag=now_list,sr=local,generated=1; tag=now_list,sr=local,generated=1,hitch=after_sources; 10*tag=now_list,sr=local,generated=1"},
    {"fg-on-hitch-real-before-generated-4x", "20*sr=local; fg=on,tag=now_list,sr=local,generated=3; 10*tag=now_list,sr=local,generated=3; tag=now_list,sr=local,generated=3,hitch=after_sources; 10*tag=now_list,sr=local,generated=3"},
    {"fg-on-hitch-after-present", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1; 10*tag=now_list,sr=local,generated=1; tag=now_list,sr=local,generated=1,hitch=after_present; 10*tag=now_list,sr=local,generated=1"},
    {"fg-on-dlssg-in-flight-hitch-after-sources", "20*sr=local; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present,hitch=after_sources; 10*sr=local,dlssg=compute,dlssg_gpu=after_present"},
    {"fg-on-live-compute-hitch-after-sources", "20*sr=local; fg=on,sr=local,dlssg=compute,generated=1; 10*sr=local,dlssg=compute,dlssg_gpu=after_present,generated=1; sr=local,dlssg=compute,dlssg_gpu=after_present,generated=1,hitch=after_sources; 10*sr=local,dlssg=compute,dlssg_gpu=after_present,generated=1"},
    {"fg-on-owned-sr-hitch-after-sources", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; sr=local,generated=1,at=evaluate,hitch=after_sources; 10*sr=local"},
    // A hitch between frames, after a frame's Presents and before the next
    // frame's first call, whose own Present reads its own copy: SR, an FG
    // tag at FG On, an OnlyValidNow tag outside FG that its SR evaluation
    // uses, and DLSS-G's input on the game's list. Until the 10-27 review the
    // judge took such a pause as a warm-up (J1).
    {"hitch-between-sr", "20*sr=local; sr=local,hitch=between; 10*sr=local"},
    {"hitch-between-fg-tag", "20*sr=local; fg=on,tag=now_list,sr=local; 5*tag=now_list,sr=local; tag=now_list,sr=local,hitch=between; 10*tag=now_list,sr=local"},
    {"hitch-between-sr-tag", "20*sr=local; 10*tag=now_list,sr=tag; tag=now_list,sr=tag,hitch=between; 10*tag=now_list,sr=tag"},
    {"hitch-between-dlssg-direct", "20*sr=local; fg=on,sr=local,dlssg=direct; 5*sr=local,dlssg=direct; sr=local,dlssg=direct,hitch=between; 10*sr=local,dlssg=direct"},
    // After a hitch between a frame's evaluation and its Present, frames
    // whose tag copy is deferred to the Present after them (R2): the first
    // can only hold the copy before the hitch, past the source age, so mono
    // is the contract there (the judge called it a gap until the 10-27
    // review, J3, and passed an over-age hold).
    {"hitch-after-sources-then-deferred-tag", "20*sr=local; sr=local,hitch=after_sources; 5*tag=present_list"},
    // A newest nomination that failed or was refused is a missing frame of
    // its display lineage (10-28): the lineage's copy completed before it
    // stands in at once and no gap watermark moves. Until then each one
    // revoked that copy queue-wide (completed_before_gap): the Present held
    // an older copy while it was readable. With FG on: a live scope's DLSS-G
    // attempt refused (compute_unobserved) after a brief hitch, its previous
    // compute copy in flight at its own Present behind a poisoned order
    // fence; then a second brief hitch let the held copy expire, mono at a
    // Present while that compute copy was readable (dropped_hold at frame 28
    // in five of five runs, 10-28, temporary build). Without FG: every other
    // SR tag on another queue unprovable (refused at its call) while each SR
    // copy is readable only after its own Present (the CPU poisons that
    // queue's order fence, as O3 does): stale depth beside readable newer
    // copies, then about a second of mono (stale_shown from frame 34,
    // 25-27 failures, in five of five runs, 10-28, temporary build).
    {"refused-fg-attempt-keeps-completed-copy", "20*sr=local; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present,order=second; 5*sr=local,dlssg=compute,dlssg_gpu=after_present; hitch=brief,sr=local,dlssg=compute,dlssg_gpu=after_present; sr=local,dlssg=compute_unobserved; hitch=brief; 10*sr=local,dlssg=compute,dlssg_gpu=after_present", true},
    {"refused-sr-tags-keep-completed-copies", "20*sr=other_queue; sr=other_queue,order=cpu; 10*sr=other_queue; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; sr=other_queue; tag=now_list,state=unprovable; "
      "10*sr=other_queue", true},
    // The first form without its hitches: the refused DLSS-G attempt's
    // previous compute copy shows at once. With the missing-frame rule
    // disabled the Present holds the copy before it, one frame (16 ms)
    // late: only the freshness rule by capture id sees it (stale_shown at
    // frame 26, capture 366 behind 371, 10-28, temporary build); the tick
    // bound passed it.
    {"refused-fg-attempt-shown-at-once", "20*sr=local; fg=on,sr=local,dlssg=compute,dlssg_gpu=after_present,order=second; 5*sr=local,dlssg=compute,dlssg_gpu=after_present; sr=local,dlssg=compute_unobserved; 10*sr=local,dlssg=compute,dlssg_gpu=after_present", true},
    // FG Off retires only the FG role of its viewport's display lineage
    // (10-28): the SR copies recorded since the shown FG copy stay its legal
    // successor. Until then FG Off revoked them. An FG owner: a silent
    // frame's SR copy on the game's queue, in flight at its own Present and
    // ordered by no fence (a second signaller), before a late Off; the next
    // frame's own SR copy is in flight too, so its Present held the FG copy
    // beside the readable SR copy (stale_shown at frame 28, the FG copy
    // after FG Off judged by its lineage, in five of five runs, 10-28,
    // temporary build). An SR owner whose newest nomination is DLSS-G's
    // copy at a late Off, the frame's SR copy held on the GPU past its
    // Present behind an order fence a CPU signal poisoned (O3): the
    // namespace's head named no view, so the Present after the Off read an
    // interruption and released the queue to Generic, then that SR copy was
    // refused (completed_before_gap): a gap at frame 24 and mono at frame
    // 25, in five of five runs, 10-28, temporary build. Its form with a
    // second signaller in place of the CPU's failed one run of eight with a
    // stale_shown at frame 22, before the Off (one Present behind a copy
    // that completed right before it; not reproduced in three replays).
    {"fg-off-late-sr-after-fg-copy", "20*sr=other_queue,sr_gpu=after_present; fg=on,tag=now_list,sr=other_queue,sr_gpu=after_present; 5*tag=now_list,sr=other_queue,sr_gpu=after_present; sr=other_queue,sr_gpu=after_present; fg=off_late,sr=other_queue,sr_gpu=after_present,order=second; sr=other_queue,sr_gpu=after_present; 10*sr=other_queue,sr_gpu=after_present", true},
    {"fg-off-late-dlssg-head-sr-owner", "20*sr=other_queue,sr_gpu=after_present; fg=on,sr=other_queue,sr_gpu=after_present; sr=other_queue,order=cpu; 2*sr=other_queue,sr_gpu=after_present; fg=off_late,sr=other_queue,sr_gpu=after_present,dlssg=direct; sr=other_queue,sr_gpu=after_present; 10*sr=other_queue,sr_gpu=after_present", true},
    // The judge's stale rule ages a Present that holds a copy from the
    // newest offered API copy's made tick, not its frame's stamp: a frame
    // with a brief hitch before its calls made its SR copy that much later
    // (until 10-28 the Present after FG Off holding that 157 ms old copy was
    // stale at 297 ms, and every isolation Present after it).
    {"judge-hold-age-from-made-tick", "fg=on,tag=now_list; sr=other_queue; sr=other_queue,hitch=brief; fg=off,hitch=brief"},
    // FG On and a hitch of the source age and more between a frame's calls
    // and its Present, the frame's FG depth a ValidUntilPresent tag without a
    // command list, whose copy is deferred to that Present (R2) and which
    // DLSS-G's own evaluation of the frame stands aside for: that Present
    // shows the frame's own copy (frame_generation::deferral_evaluations,
    // expired). Until 10-28 the deferral lapsed by the clock at its own
    // Present, so the frame had no FG copy and its Present was mono
    // (FG_scope_required, inactive_views): open item A, seed
    // 2d28df102b516f4d at frame 49 and the fixed set's c28965dca8d7c76c at
    // frame 64, which the judge passed as a stall's warm-up. Forms: DLSS-G on
    // the game's list, the viewport's SR owned by the scope (the seed's
    // shape), DLSS-G's compute copy, 4x, the generated Present inside the
    // recording or inside the SR evaluation, and the tag alone.
    {"fg-on-deferred-tag-hitch-after-sources", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=1; 5*tag=present_bare,dlssg=direct,generated=1; tag=present_bare,dlssg=direct,hitch=after_sources; 10*tag=present_bare,dlssg=direct,generated=1"},
    {"fg-on-deferred-tag-hitch-after-sources-owned-sr", "20*sr=local; fg=on,tag=present_bare,sr=local,dlssg=direct,generated=1; 5*tag=present_bare,sr=local,dlssg=direct,generated=1; tag=present_bare,sr=local,dlssg=direct,hitch=after_sources; 10*tag=present_bare,sr=local,dlssg=direct,generated=1"},
    {"fg-on-deferred-tag-hitch-after-sources-dlssg-compute", "20*sr=local; fg=on,tag=present_bare,dlssg=compute,generated=1; 5*tag=present_bare,dlssg=compute,generated=1; tag=present_bare,dlssg=compute,hitch=after_sources; 10*tag=present_bare,dlssg=compute,generated=1"},
    {"fg-on-deferred-tag-hitch-after-sources-4x", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=3; 5*tag=present_bare,dlssg=direct,generated=3; tag=present_bare,dlssg=direct,generated=3,hitch=after_sources; 10*tag=present_bare,dlssg=direct,generated=3"},
    {"fg-on-deferred-tag-hitch-after-sources-inside", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=1,at=inside; 5*tag=present_bare,dlssg=direct,generated=1,at=inside; tag=present_bare,dlssg=direct,generated=1,at=inside,hitch=after_sources; 10*tag=present_bare,dlssg=direct,generated=1,at=inside"},
    {"fg-on-deferred-tag-hitch-after-sources-evaluate", "20*sr=local; fg=on,tag=present_bare,sr=local,dlssg=direct,generated=1,at=evaluate; 5*tag=present_bare,sr=local,dlssg=direct,generated=1,at=evaluate; tag=present_bare,sr=local,dlssg=direct,generated=1,at=evaluate,hitch=after_sources; 10*tag=present_bare,sr=local,dlssg=direct,generated=1,at=evaluate"},
    {"fg-on-deferred-tag-hitch-after-sources-tag-only", "20*sr=local; fg=on,tag=present_bare,generated=1; 5*tag=present_bare,generated=1; tag=present_bare,hitch=after_sources; 10*tag=present_bare,generated=1"},
    {"fg-on-deferred-tag-hitch-after-sources-tag-only-evaluate", "20*sr=local; fg=on,tag=present_bare,sr=local,generated=1,at=evaluate; 5*tag=present_bare,sr=local,generated=1,at=evaluate; tag=present_bare,sr=local,generated=1,at=evaluate,hitch=after_sources; 10*tag=present_bare,sr=local,generated=1,at=evaluate"},
    // The same hitch after frames whose generated Present DLSS-G skipped
    // (FG 2x declared, none presented): the hitch frame's real Present
    // counts as the earlier frame's missing generated Present while no later
    // evaluation began (frame_presenting). Until 10-28 that frame's lapsed
    // deferral began none, so its Present showed the earlier frame's copy,
    // past the source age, as current (stale). A hitch frame that makes no
    // FG call, its SR evaluation owned by the live scope, ends that window
    // too (depth_capture::evaluation_called): its Present may be the one
    // flat Present "Leave it" accepts, never the earlier frame's copy.
    {"fg-on-deferred-tag-missing-generated-hitch", "20*sr=local; fg=on,tag=present_bare; 5*tag=present_bare,dlssg=direct; tag=present_bare,dlssg=direct,hitch=after_sources; 10*tag=present_bare,dlssg=direct"},
    // A late FG Off whose frame's FG depth is a deferred tag copy (R2),
    // made at the frame's Present after the Off: the copy is that frame's
    // own depth of the lineage, the viewport's ordinary source, and the next
    // frame's readable SR copy replaces it (the 10-28 review's P1, shrunk
    // to its first form). Until 10-28 the copy became the viewport's FG
    // view again beside its SR view, so every Present after the Off read
    // two active views (ambiguous_views), revoked each readable SR copy and
    // held the stale FG copy until it expired: stale_shown at every frame
    // after the Off (10 of 10), then mono.
    {"fg-off-late-deferred-tag-copy", "fg=on; fg=off_late,tag=present_bare; sr=local"},
    {"fg-off-late-deferred-tag-copy-sr-after", "20*sr=local; fg=on,tag=present_bare,sr=local; 5*tag=present_bare,sr=local; fg=off_late,tag=present_bare,sr=local; 10*tag=present_bare,sr=local"},
    // Seed dda773a575be6561's frames 80 to 91 before seeds drew late Offs
    // (10-28), after SR on the game's queue has run every entry of its ring:
    // FG renewed (its fences anew), a brief hitch, DLSS-G's refused last
    // evaluation, then FG Off with the frame's own OnlyValidNow tag copy on
    // the presenting queue while its SR copy on the game's queue waits for a
    // renewed fence. The FG Off frame shows its own tag copy. Once in a
    // fixed-set run of the 10-28 review (J2), and once in the fixed set of
    // a full check (10-28), it held a 93 ms old FG copy there instead
    // (stale_shown; the add-on's FG switch line named completed_not_older):
    // the frame's SR copy completed between the selection's two looks at
    // it, so the head itself was its newest completed copy. In isolation
    // (25 runs) it did not reproduce; the seam characterization pins it.
    {"fg-off-own-tag-copy-after-dlssg-refusal", "20*sr=other_queue; fg=on,tag=now_list,dlssg=direct,generated=2; 2*tag=now_list,dlssg=direct,generated=2; 2*dlssg=direct,generated=2; sr=other_queue,dlssg=direct,generated=2,order=wait_first; fg=on,sr=other_queue,dlssg=direct,generated=2,order=wait_first; sr=other_queue,dlssg=direct,generated=2,order=wait_first,hitch=brief; sr=other_queue,dlssg=direct,generated=2,order=wait_first; untag=after_global,sr=other_queue,dlssg=direct_unobserved,generated=2,order=wait_first; fg=off,tag=now_list,untag=after_global,sr=other_queue,order=wait_first; tag=now_list,untag=before_global,sr=other_queue,order=wait_first; 10*sr=other_queue"},
    {"fg-on-owned-sr-missing-generated-hitch", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 5*tag=now_list,sr=local; sr=local,hitch=after_sources; 10*sr=local"},
    // FG On, a hitch of the source age and more after a frame's real
    // Present (after_present) or before the next frame's first call
    // (between), then the next frame's FG tag call, whose copy is deferred to
    // that frame's Present (R2), and DLSS-G's generated Present of the
    // earlier frame inside the next frame's recording (at=inside). The
    // deferred tag call begins no evaluation, so that generated Present
    // still presents the earlier frame and shows its own copy, however old
    // (frame_presenting; owner ruling Q2, 10-30: no 2D flash there). Until
    // 10-31 the judge counted the tag call's decision as the next frame's
    // first evaluation and judged that copy stale, every run. Forms: seed
    // d12563f8dc22b059's frames 78 and 79 (the earlier frame's tag copied at
    // its SR evaluation of it, the hitch after its Present), the bare tag
    // with the hitch between frames, and seed 2dbc7936fb23f7fa's frames 86
    // to 93 (a frame tag with a global untag after it, R7's EITHER; the
    // earlier frame's SR copy on the game's queue).
    {"fg-on-deferred-tag-generated-inside-after-hitch", "20*sr=local; fg=on,tag=evaluate_list,entry=global,sr=tag,dlssg=direct,generated=1,at=inside; 5*tag=evaluate_list,entry=global,sr=tag,dlssg=direct,generated=1,at=inside; tag=evaluate_list,entry=global,sr=tag,dlssg=direct,generated=1,at=inside,hitch=after_present; 10*tag=present_list,entry=global,sr=tag,dlssg=direct,generated=1,at=inside"},
    {"fg-on-deferred-tag-generated-inside-after-hitch-between", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=1,at=inside; 5*tag=present_bare,dlssg=direct,generated=1,at=inside; tag=present_bare,dlssg=direct,generated=1,at=inside,hitch=between; 10*tag=present_bare,dlssg=direct,generated=1,at=inside"},
    {"fg-on-deferred-tag-generated-inside-after-hitch-untag", "20*sr=local; fg=on,sr=local,dlssg=direct,generated=2; 2*tag=evaluate_list,untag=after_global,sr=local,dlssg=direct,generated=2; tag=evaluate_list,untag=after_global,dlssg=direct,generated=2,ui=bare; tag=evaluate_list,untag=after_global,generated=2,at=inside,ui=bare; tag=present_bare,untag=after_global,sr=other_queue,generated=2,at=inside,ui=bare; tag=present_bare,untag=after_global,sr=other_queue,generated=2,at=inside,ui=bare,hitch=between; tag=present_bare,untag=before_global,generated=2,at=inside,ui=untagged; 10*tag=present_bare,ui=untagged"},
    // FG 4x declared and DLSS-G presenting no generated frame, each frame's
    // FG depth a tag copy deferred to its Present (R2), then two frames whose
    // global untag ends that copy, so the scope falls silent, and DLSS-G's
    // own compute copy of each refused (no observed state), the second after
    // a hitch of the source age between frames. Its real Present does not
    // present the frame before the untags: DLSS-G's evaluation of each later
    // frame ends that frame's window (frame_presenting) whatever its outcome,
    // so the hitch frame holds nothing past the source age. Until 10-31 a
    // silent scope's refused DLSS-G attempt began no evaluation, so that
    // Present showed the earlier frame's copy, 453 ms old, as its own (stale
    // at frame 27, every run). Soak seed 4547e4874aef62bb's frames 73 to 81
    // failed so only under CPU load: its DLSS-G copies about a second
    // earlier kept the scope live at the untag (miss_frame_generation) while
    // they were younger than fg_silent_ms, and a live scope's refused
    // attempt begins its nomination.
    {"fg-on-silent-dlssg-refusal-hitch-between", "20*sr=local; fg=on,tag=present_bare,generated=3; 4*tag=present_bare; tag=present_bare,dlssg=compute_unobserved; tag=present_bare,untag=after_global,dlssg=compute_unobserved; tag=present_bare,untag=after_global,dlssg=compute_unobserved,hitch=between; 10*tag=present_bare,untag=after_global,sr=local,dlssg=compute_unobserved"},
    // MFG with the earlier frame's generated Presents inside this frame's
    // recording after its first stamp (DLSS-G's present thread: Assassin's
    // Creed Shadows), inside its SR evaluation (at=evaluate) at 2x, 3x and
    // 4x, or after its FG tag copy and before its SR evaluation (at=inside),
    // then a hitch of the source age before its real Present, and its own
    // generated Presents inside the next frame's recording before any call.
    // Its presenting window is anchored at its real Present, the first after
    // its copy's submission (frame_presenting), so those Presents hold its
    // own copy and its SR evaluation after the earlier Presents does not end
    // its window. Until 10-31 the window counted from its stamp, the earlier
    // frame's Presents used it up, and its own generated Presents after the
    // freeze were mono (dropped_hold at frame 32, every run; soak seed
    // d40ba4d5b7c456ee at frame 126).
    {"fg-on-generated-in-evaluate-hitch-before-real-2x", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 10*tag=now_list,sr=local,generated=1,at=evaluate; tag=now_list,sr=local,generated=1,at=evaluate,hitch=after_sources; sr=local,generated=1,at=inside; 10*sr=local"},
    {"fg-on-generated-in-evaluate-hitch-before-real-3x", "20*sr=local; fg=on,tag=now_list,sr=local,generated=2,at=evaluate; 10*tag=now_list,sr=local,generated=2,at=evaluate; tag=now_list,sr=local,generated=2,at=evaluate,hitch=after_sources; sr=local,generated=2,at=inside; 10*sr=local"},
    {"fg-on-generated-in-evaluate-hitch-before-real-4x", "20*sr=local; fg=on,tag=now_list,sr=local,generated=3,at=evaluate; 10*tag=now_list,sr=local,generated=3,at=evaluate; tag=now_list,sr=local,generated=3,at=evaluate,hitch=after_sources; sr=local,generated=3,at=inside; 10*sr=local"},
    {"fg-on-generated-before-own-evaluation-hitch-before-real-4x", "20*sr=local; fg=on,tag=now_list,sr=local,generated=3,at=inside; 10*tag=now_list,sr=local,generated=3,at=inside; tag=now_list,sr=local,generated=3,at=inside,hitch=after_sources; sr=local,generated=3,at=inside; 10*sr=local"},
    // FG Off after a silent FG frame's SR copy on the game's queue, kept
    // off-screen by the FG hold, then two frames refused (no observed
    // state): the display holds the last FG tag copy (deferred to its
    // Present) and the lineage's SR successor replaces it only with a copy
    // published after it. Until 10-31 FG Off at frame 31 showed that SR
    // copy, two frames older than the held FG copy, and held it until the
    // next SR copy (about 125 ms), every run; the judge saw the step back
    // only since it judges the same API too (seed c40a42f921da5ba0, frames
    // 138 to 162, the deferred list's item 11).
    {"fg-off-sr-successor-older-than-held", "20*sr=other_queue,sr_gpu=after_present; 3*tag=present_bare,ui=listed; fg=on,tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; 2*tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,ui=listed; 2*tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; fg=off,tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; 2*sr=other_queue,state=unprovable,ui=listed,order=late; sr=other_queue,state=unprovable,order=late; fg=off,sr=other_queue,state=unprovable,order=late; tag=now_list,sr=other_queue,state=unprovable,order=late,sr_gpu=after_present; tag=evaluate_bare,sr=other_queue,state=unprovable,ui=listed,order=late; tag=evaluate_bare,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed,order=late; 3*tag=evaluate_bare,state=proven,ui=listed; 2*ui=listed"},
    // FG Off right after a frame whose only FG input is DLSS-G's compute
    // copy, in flight at its own Present (the first after FG On renewed the
    // game's order fence, or dlssg_gpu=after_present), so that Present held
    // the frame before's copy. FG Off keeps the scope's copies published
    // after the shown copy as the lineage's ordinary copies, so the compute
    // copy, readable from the next Present, replaces the held one. Until
    // 10-31 FG Off invalidated it: the display held the older copy beside
    // it (stale_shown at frames 31 to 33, and 29 to 32 with dynamic
    // resolution, every run; seeds 40e5fd9a22d273bc at frames 107 to 109 and
    // 64863a396a5d6dd5 at frames 139 to 142), or, after a late Off with the
    // display on the silent scope's SR copy, the queue lost its association
    // and the Present went mono (dropped_hold at frame 27, every run; seed
    // 117a6b99131447b1 at frame 57, in about one run of three).
    {"fg-off-late-pending-dlssg-copy", "20*sr=local; 3*tag=present_bare,sr=local,generated=2; drs=change,tag=present_bare,sr=local,generated=2; tag=present_bare,sr=local,generated=2; tag=present_bare,sr=local,generated=2,at=inside; fg=on,tag=present_bare,sr=local,generated=2,at=inside; drs=change,tag=present_bare,sr=local,generated=2,at=inside; tag=present_bare,sr=local,generated=1,at=inside; tag=present_bare,sr=local,state=unprovable,generated=3,at=inside; tag=present_bare,sr=local,dlssg=compute,state=unprovable,generated=3,at=inside,dlssg_gpu=after_present; fg=off_late,tag=present_bare,sr=local,dlssg=compute,state=unprovable; 2*tag=present_bare,sr=local,state=unprovable; tag=present_bare,sr=local,state=proven; fg=off,tag=present_bare,sr=local,state=unprovable; tag=present_bare,sr=local,state=unprovable; tag=present_bare,state=unprovable; drs=change,tag=present_bare,state=proven"},
    {"fg-off-drs-pending-dlssg-copy", "20*sr=local; fg=on,tag=present_list,sr=tag; 4*tag=present_list,sr=tag; 3*tag=present_list; untag=before_global,dlssg=compute,dlssg_gpu=after_present; fg=off,drs=change; 3*defaults; fg=off,sr=local; 6*tag=now_list,sr=local"},
    {"fg-off-late-pending-dlssg-copy-silent-scope", "20*tag=evaluate_list; fg=off,tag=evaluate_list; tag=evaluate_list; 2*tag=evaluate_list,untag=after_frame; fg=on,tag=evaluate_list,untag=after_frame,sr=other_queue,dlssg=direct_unobserved,generated=1,at=inside; drs=change,tag=evaluate_list,untag=after_frame,sr=other_queue,dlssg=direct_unobserved,generated=1,at=inside; tag=evaluate_list,untag=after_frame,dlssg=compute,generated=1,at=inside,dlssg_gpu=after_present; fg=off_late,tag=evaluate_list,dlssg=compute; tag=evaluate_list; tag=evaluate_list,ui=untagged; tag=evaluate_bare,ui=untagged; ui=untagged; 2*untag=after_frame,ui=untagged; untag=before_global,ui=untagged"},
    // A frame's FG tag copy deferred to its Present (R2), with a hitch of
    // the source age between the frame's calls and that Present (a game
    // thread preempted between them, for less), then refused frames through
    // FG Off until the next SR copy, as in fg-off-sr-successor-older-than-held
    // (seed c40a42f921da5ba0's frames 138 to 162; its refused frames without
    // their untags, which would let them drop the hold): the display holds
    // that copy, aged from the Present that made it, as the judge ages it.
    // Until 11-01 the add-on aged it from its tag call, so the first refused
    // frame after the hitch was mono (dropped_hold at frame 29, every run;
    // with the seed's untags the judge excused the same mono Presents).
    {"fg-on-deferred-tag-hitch-after-sources-held", "20*sr=other_queue,sr_gpu=after_present; 3*tag=present_bare,ui=listed; fg=on,tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; 2*tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=proven,ui=listed,hitch=after_sources; tag=present_bare,sr=other_queue,state=unprovable,ui=listed; tag=present_bare,sr=other_queue,state=unprovable,ui=listed,order=second; fg=off,tag=present_bare,sr=other_queue,state=unprovable,ui=listed,order=second; sr=other_queue,state=unprovable,ui=listed,order=second; 2*sr=other_queue,state=unprovable,ui=listed,order=late; sr=other_queue,state=unprovable,order=late; fg=off,sr=other_queue,state=unprovable,order=late; tag=now_list,sr=other_queue,state=unprovable,order=late,sr_gpu=after_present; tag=evaluate_bare,sr=other_queue,state=unprovable,ui=listed,order=late; tag=evaluate_bare,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed,order=late; 3*tag=evaluate_bare,state=proven,ui=listed; 2*ui=listed"},
    // A late FG Off whose frame's DLSS-G copy, made at its evaluation, is
    // kept as the viewport's ordinary source and shown at that frame's
    // Present after a hitch since the evaluation (fixed-set seed
    // 5d48a3d414f1de10's frames 0 to 40, with hitch=after_sources on its
    // fg=off_late frame 33). The fixture supplies that copy as permitted,
    // aged from its evaluation as the add-on ages it. Until 11-01 it
    // stripped every FG copy of a late-Off frame but its deferred tag copy,
    // so the shown copy failed as unsupplied (oracle:unsupplied_fresh at
    // frame 33, every run), and without the hitch the add-on's hold of it
    // was aged from the frame before (the seed, stale at frame 39 about 1
    // run in 3).
    {"fg-off-late-kept-dlssg-copy-hitch-after-sources", "sr=other_queue,state=recorded_other,order=cpu; tag=evaluate_bare,state=recorded_other; tag=evaluate_bare,state=recorded_other; tag=evaluate_bare,state=recorded_other; tag=evaluate_bare,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; sr=local,state=recorded_other; untag=before_global,sr=local,state=recorded_other; fg=on,untag=before_global,sr=other_queue,state=recorded_other; untag=before_global,dlssg=direct_unobserved,state=proven; untag=before_global,dlssg=direct,state=proven; untag=before_global,dlssg=direct,state=proven; untag=before_global,dlssg=direct,state=recorded_other; untag=before_global,dlssg=direct; untag=before_global,dlssg=direct; untag=before_global,dlssg=direct; untag=before_global,dlssg=direct,generated=2; untag=before_global,dlssg=direct,state=unprovable,generated=2; untag=before_global,generated=2; untag=after_frame,generated=2; untag=after_frame,generated=2; untag=before_global,generated=2; untag=before_global,generated=2; untag=before_global,dlssg=direct,generated=2; untag=before_global,dlssg=direct,generated=2; tag=now_bare,untag=before_global,dlssg=direct,generated=2; tag=now_bare,untag=before_global,dlssg=direct,generated=2; tag=now_bare,untag=before_global,dlssg=direct,generated=2; fg=off_late,tag=now_bare,entry=global,untag=before_global,sr=tag,dlssg=direct,hitch=after_sources; tag=now_bare,entry=global,untag=before_global,sr=tag; drs=change,tag=now_bare,entry=global,untag=before_global,sr=tag; tag=now_bare,entry=global,untag=before_global,sr=tag,ui=untagged; tag=now_bare,entry=global,untag=before_global,sr=tag,ui=untagged,hitch=brief; tag=now_bare,entry=global,sr=tag,ui=untagged; tag=now_bare,entry=global,sr=tag,ui=untagged; tag=now_bare,entry=global,sr=tag,ui=untagged"},
    // fg-on-deferred-tag-hitch-after-sources-held with untags on its four
    // refused frames after the hitch (seed c40a42f921da5ba0's own shape),
    // in either form. An untag before the frame's tag call ends nothing, as
    // that call names the depth again, so those Presents must hold the hitch
    // frame's deferred copy as without the untags (the fourth frame has no
    // tag, so its untag may end the hold in either form); an untag after the
    // tag (R7) may end the source's hold, so a mono Present there passes. Until
    // 11-01 every untag form excused the drop: with the add-on aging that
    // copy from its tag call again (a temporary mutation of
    // present_deferred), both forms went mono from frame 29 and passed; the
    // before form now fails there (dropped_hold), the after form passes.
    {"fg-on-deferred-tag-hitch-after-sources-held-untag-before", "20*sr=other_queue,sr_gpu=after_present; 3*tag=present_bare,ui=listed; fg=on,tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; 2*tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=proven,ui=listed,hitch=after_sources; tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; fg=off,tag=present_bare,untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; untag=before_global,sr=other_queue,state=unprovable,ui=listed,order=second; 2*sr=other_queue,state=unprovable,ui=listed,order=late; sr=other_queue,state=unprovable,order=late; fg=off,sr=other_queue,state=unprovable,order=late; tag=now_list,sr=other_queue,state=unprovable,order=late,sr_gpu=after_present; tag=evaluate_bare,sr=other_queue,state=unprovable,ui=listed,order=late; tag=evaluate_bare,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed,order=late; 3*tag=evaluate_bare,state=proven,ui=listed; 2*ui=listed"},
    {"fg-on-deferred-tag-hitch-after-sources-held-untag-after", "20*sr=other_queue,sr_gpu=after_present; 3*tag=present_bare,ui=listed; fg=on,tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; tag=present_bare,sr=other_queue,sr_gpu=after_present,dlssg=direct,state=recorded_other,ui=listed; 2*tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed; tag=present_bare,untag=before_global,sr=other_queue,state=proven,ui=listed,hitch=after_sources; tag=present_bare,untag=after_global,sr=other_queue,state=unprovable,ui=listed; tag=present_bare,untag=after_global,sr=other_queue,state=unprovable,ui=listed,order=second; fg=off,tag=present_bare,untag=after_global,sr=other_queue,state=unprovable,ui=listed,order=second; untag=after_global,sr=other_queue,state=unprovable,ui=listed,order=second; 2*sr=other_queue,state=unprovable,ui=listed,order=late; sr=other_queue,state=unprovable,order=late; fg=off,sr=other_queue,state=unprovable,order=late; tag=now_list,sr=other_queue,state=unprovable,order=late,sr_gpu=after_present; tag=evaluate_bare,sr=other_queue,state=unprovable,ui=listed,order=late; tag=evaluate_bare,sr=other_queue,sr_gpu=after_present,state=proven,ui=listed,order=late; 3*tag=evaluate_bare,state=proven,ui=listed; 2*ui=listed"},
    // A hitch of the source age right before DLSS-G's own evaluation of a
    // frame whose FG depth is a tag copy deferred to its Present (a stall
    // before the game's Present call, inside which DLSS-G evaluates):
    // DLSS-G no longer stands aside for that tag copy (its tag call is past
    // the source age), its own copy is refused (its input's state
    // unobserved), and that refused attempt ends an earlier frame's
    // presenting window (depth_capture::evaluation_called, 10-31), not this
    // frame's own: its Present shows its deferred copy, made there, and its
    // generated Presents hold it, 2x and 4x. A guard: round 1 left this
    // shape unmodelled (hitch=after_sources comes after DLSS-G's
    // evaluation).
    {"fg-on-deferred-tag-hitch-before-dlssg-refused", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=1; 5*tag=present_bare,dlssg=direct,generated=1; tag=present_bare,dlssg=direct_unobserved,generated=1,hitch=before_dlssg; 10*tag=present_bare,dlssg=direct,generated=1"},
    {"fg-on-deferred-tag-hitch-before-dlssg-refused-4x", "20*sr=local; fg=on,tag=present_bare,dlssg=direct,generated=3; 5*tag=present_bare,dlssg=direct,generated=3; tag=present_bare,dlssg=direct_unobserved,generated=3,hitch=before_dlssg; 10*tag=present_bare,dlssg=direct,generated=3"},
    // A frame's FG tag copy deferred to its Present with MFG 4x, the earlier
    // frame's generated Presents inside its recording and a hitch of the
    // source age after its calls, while the add-on's metadata reads fail at
    // its Presents (PD5, lane P 11-01): present_deferred took the FG mode
    // from the camera records, and a lost read counted as generating with no
    // generated Presents, so the deferral expired at its own real Present
    // (three Presents after its tag call) and lapsed unmade. The FG mode now
    // comes from the FG options store, which no read loses.
    {"fg-on-lost-metadata-read-hitch", "20*sr=local; fg=on,tag=present_bare,generated=3,at=inside; 5*tag=present_bare,generated=3,at=inside; tag=present_bare,generated=3,at=inside,hitch=after_sources,reads=lost_at_present; 10*tag=present_bare,generated=3,at=inside"},
    // FG turned on, and later off, by an options call inside a frame's real
    // Present, after the UI read that Present's FG mode and before depth
    // read it (PD11, lane P 11-01): until one FG snapshot per Present that
    // Present's UI used the old mode and its depth the new one
    // (fg_modes:disagree at frame 20, and at frame 31); now both use the
    // snapshot taken at the Present's start, and the next Present the new
    // mode.
    {"fg-on-between-ui-and-depth-reads", "20*sr=local; fg=on,fg_at=depth_read,sr=local; 20*sr=local"},
    {"fg-off-between-ui-and-depth-reads", "20*sr=local; fg=on,sr=local; 10*sr=local; fg=off,fg_at=depth_read,sr=local; 10*sr=local"},
    // The UI's generated-Present hold ends where depth's window does, at the
    // next frame's evaluation (PD7, lane P 11-01). FG on, no generated
    // Presents (DLSS-G skipped them), UI tags every frame; frame 24 freezes
    // before its sources, so its own Present offers no UI input (the last
    // snapshot aged past 250 ms, its own deferred one is not made yet).
    // Until then the UI counted one Present from frame 23's and held frame
    // 23's decision on frame 24's real Present as its generated Present
    // (ui_frame:held at frame 24); now depth's window, which frame 24's
    // evaluation ended, decides, and the T1 grace reuses it as a real
    // frame without its own decision.
    {"fg-on-ui-window-ends-at-next-evaluation-hitch", "20*sr=local; fg=on,sr=local,dlssg=direct; 3*sr=local,dlssg=direct,ui=listed; sr=local,dlssg=direct,ui=bare,hitch=before_sources; 5*sr=local,dlssg=direct,ui=listed"},
    // A conflict of the shown copy's read (O1) while the frame's own copy
    // is in flight, beside an earlier frame's copy whose capture id the
    // fixture never learned (the add-on published none it saw at the call)
    // and that is younger than the source age: the add-on drops the
    // conflicted copy and that Present is mono, as O1 has it, since the
    // earlier copy was published before the shown one (it never steps
    // back). Until 11-01 the judge counted any copy without an id as
    // holdable, so that mono Present was a gap (every run): the first
    // DLSS-G compute copy after FG On, read through no fence (its fence
    // not yet settled), then a second signaller of that fence (soak seed
    // 2bda274f7f2ca5de at frame 63, in loaded runs only), and the FG frame's
    // SR copy before FG Off, then a late signal (seed afcf3faa4ef87af5's
    // frames 0 to 12, at frame 12 in loaded runs only). The reads are held
    // on the GPU past their Presents, so each run makes the late read.
    {"O1-second-signal-unknown-copy-of-earlier-frame", "20*sr=local; fg=on,tag=present_list,sr=local; tag=present_list,sr=local; 4*tag=present_list,sr=local,dlssg=compute; tag=now_bare,sr=local,dlssg=compute,dlssg_gpu=after_present; tag=now_bare,sr=local,dlssg=compute,dlssg_gpu=after_present,order=second; 10*tag=now_bare,sr=local,dlssg=compute", false, 1},
    {"O1-late-signal-unknown-copy-of-earlier-frame", "tag=present_list,sr=tag,ui=bare; tag=present_list,sr=tag; 6*tag=present_list,sr=other_queue; tag=present_list,sr=other_queue,ui=listed; fg=on,tag=present_list,sr=other_queue,dlssg=compute_unobserved,state=proven,generated=1,ui=listed; fg=off,sr=other_queue,state=proven,ui=listed,sr_gpu=after_present; sr=other_queue,state=proven,ui=listed,order=late,sr_gpu=after_present; sr=other_queue,state=proven,ui=listed,order=late,sr_gpu=after_present; 10*sr=other_queue,state=proven,ui=listed", false, 1},
    // Generic's copy shown at the Present that made it (its end-of-frame
    // copy) is the presented frame's own depth, however long the frame took:
    // a hitch of the source age between the frame's calls and its Present.
    // Until 11-01 the judge aged it from the frame's stamp, so that Present
    // was stale (frame 20, every run; soak seed 543ed82ea2b4170f at frame 0,
    // in the soak's process only, where Generic was ready at the first
    // Present; alone it was not, and the first Present was mono).
    {"generic-own-copy-hitch-after-sources", "20*defaults; hitch=after_sources; 10*defaults"},
    // A silent FG scope's refused FG tag call ends an earlier frame's
    // presenting window (depth_capture::evaluation_called, 11-01): soak seed
    // f0b594e3ad951a96's frames 91 to 99 and four more like 99. FG turned on
    // without an FG copy (the scope silent, each FG tag call refused at the
    // call: boundary_without_commands, then missing_state), a generated
    // Present inside each frame's recording, UI tags at frames 20 and 25 to
    // 27, untagged from frame 28. Frame 28's generated Present comes after
    // its refused tag call, so it presents frame 28 (the oracle's
    // evaluation_began). Until 11-01 that refusal ended no window, so depth
    // still presented frame 27 there and the UI held frame 27's decision as
    // a generated Present of it (ui_frame:held at frame 28, 3 of 3 runs).
    {"fg-silent-refused-tag-ends-window", "20*sr=other_queue,state=proven; tag=present_bare,untag=after_global,sr=other_queue,state=proven,ui=listed; tag=present_bare,untag=after_global,sr=other_queue,state=proven; fg=on,untag=after_global,sr=other_queue,state=proven,generated=1,at=inside,sr_gpu=after_present; 2*tag=now_bare,untag=after_frame,sr=other_queue,generated=1,at=inside; 2*tag=evaluate_list,untag=after_frame,sr=other_queue,generated=1,at=inside,ui=listed; tag=evaluate_list,untag=after_frame,sr=other_queue,generated=1,at=inside,ui=listed,sr_gpu=after_present; 5*tag=evaluate_list,untag=after_frame,sr=local,generated=1,at=inside,ui=untagged"},
    // Lane L-fg (F2, 11-01): FG 4x declared; frame 26 presents none of its
    // generated Presents, then frame 27's FG depth tag defers its copy to its
    // Present (R2), where it is refused (no provable state), after a hitch of
    // the source age between frame 27's calls and that Present, with no SR
    // or DLSS-G evaluation of frame 27 before it. The refused copy is still
    // frame 27's call at its own Present and ends frame 26's presenting
    // window. Until 11-01 nothing ended it there, so frame 27's Present held
    // frame 26's copy, 421 ms old, as its own: stale at frame 27, 2 of 2 runs.
    {"fg-on-refused-deferred-copy-ends-window-hitch", "20*sr=local; fg=on,tag=present_bare,generated=3; 5*tag=present_bare,generated=3; tag=present_bare; tag=present_bare,state=unprovable,hitch=after_sources,generated=3; 5*tag=present_bare,generated=3"},
    // The UI's generated-Present hold where it is due (ui_frame:dropped, 11-01
    // final review F13): FG 3x with DLSS-G's copy and UI tags in every frame,
    // three generated Presents after each real one, then three inside each
    // next frame's recording before its SR evaluation: each presents the
    // frame of the last UI tag's Present, so it holds that Present's UI
    // decision, where the capture owner re-offers that frame's snapshot by
    // its identity (same_snapshot_inputs). Two frames pause before their
    // calls (hitch=before_sources), so the earlier frame's snapshot is older
    // than the 250 ms the owner offers it, while the frame's own UI tag call
    // keeps the UI scope fresh: those generated Presents offer nothing and
    // hold by depth's window (generated_present, PD7). Until 11-01 only a
    // hold of the wrong frame was judged, so an add-on that dropped these
    // holds (a depth window that ended early, or a UI that compared the
    // wrong window: HUD parallax or mask flicker on every generated frame)
    // passed; it must judge at least 6 holds by the window (with the UI's
    // window hold removed, no hold wherever depth named the input's frame,
    // those 6 fail ui_frame:dropped; a UI that fell back to Present
    // counting would still hold them, which ui_frame:held judges in
    // fg-on-ui-window-ends-at-next-evaluation-hitch).
    {"fg-on-ui-generated-hold-kept", "20*sr=local; fg=on,sr=local,dlssg=direct,generated=3; 10*sr=local,dlssg=direct,generated=3,ui=listed; 5*sr=local,dlssg=direct,generated=3,at=inside,ui=bare; sr=local,dlssg=direct,generated=3,at=inside,ui=bare,hitch=before_sources; 3*sr=local,dlssg=direct,generated=3,at=inside,ui=bare; sr=local,dlssg=direct,generated=3,at=inside,ui=bare,hitch=before_sources; 3*sr=local,dlssg=direct,generated=3,at=inside,ui=bare", false, 0, 6},
    // A real Present without frame identity (11-01, OPEN C): soak seed
    // 117a6b99131447b1's frames 29 to 42, then its frames 37 to 42 again.
    // FG on with a silent scope (each FG tag call refused at the call:
    // boundary_without_commands), UI tags every frame copied at their
    // frame's Present (ui=bare, the FG inputs that stamp a frame), a tag-only
    // frame (its FG tag refused, no SR; declared one generated Present,
    // DLSS-G presents none), then a frame whose only calls before its real
    // Present, after a hitch of the source age, are its constants and its UI
    // tags (the UI's capture request lapsed in the pause, so they copy
    // nothing). The tag-only frame's refused call ends the earlier window
    // and begins its own (fg-silent-refused-tag-ends-window), which presents
    // the next Present as its generated one: the UI holds its decision there
    // (ui_unidentified; the same hold by Present count before 11-01, when
    // that frame had no window). Before the judge left that hold unjudged it
    // failed ui_frame:held at frame 31, every run; it must make at least one
    // such hold. Its frame 23's SR copy on the game's queue keeps the prompt
    // form: run after its frame's Present (sr_gpu=after_present), the parent
    // judge passed the row (REPAIR4's c1pin-old).
    {"fg-silent-tag-only-then-callless-real-present-hitch", "20*sr=local; fg=on,sr=local,state=proven,generated=1; sr=local,state=proven,generated=3; tag=evaluate_bare,sr=local,generated=2,ui=bare; tag=evaluate_bare,sr=other_queue,state=unprovable,generated=2,ui=bare; tag=now_bare,sr=local,state=proven,generated=2,at=evaluate,ui=bare; 5*tag=now_bare,sr=local,state=proven,generated=1,at=evaluate,ui=bare; tag=now_bare,state=proven,ui=bare; ui=bare,hitch=between; generated=1,ui=bare; ui=bare; 2*tag=now_bare,sr=local,state=proven,generated=1,at=evaluate,ui=bare; tag=now_bare,state=proven,ui=bare; ui=bare,hitch=between; generated=1,ui=bare; ui=bare", false, 0, 0, 1},
    // The same limit at the callless frame's own generated Presents (the
    // second fix review of 11-01): the tag-only frame declares two generated
    // Presents and DLSS-G presents none, then the callless frame after the
    // hitch presents two after its real Present. Its real Present and its
    // first generated one are both within the earlier frame's count, so
    // depth's window presents that frame at both and the UI holds its
    // decision at both; until then only the real Present was left out, and
    // the first generated one failed ui_frame:held.
    {"fg-silent-tag-only-then-callless-generated-hitch", "20*sr=local; fg=on,sr=local,state=proven,generated=1; sr=local,state=proven,generated=3; tag=evaluate_bare,sr=local,generated=2,ui=bare; tag=evaluate_bare,sr=other_queue,sr_gpu=after_present,state=unprovable,generated=2,ui=bare; tag=now_bare,sr=local,state=proven,generated=2,at=evaluate,ui=bare; 5*tag=now_bare,sr=local,state=proven,generated=2,at=evaluate,ui=bare; tag=now_bare,state=proven,ui=bare; ui=bare,hitch=between,generated=2; ui=bare; 2*tag=now_bare,sr=local,state=proven,generated=2,at=evaluate,ui=bare; tag=now_bare,state=proven,ui=bare; ui=bare,hitch=between,generated=2; ui=bare", false, 0, 0, 2},
    // The FG hold's window from the add-on's stamp of a deferred tag copy
    // (11-02): fixed-set seed dda773a575be6561's frames 140 to 175 after 20
    // SR frames. Frame 42 (the seed's 162) supplies the FG scope its tag
    // copy deferred to its Present (R2), which the silent scope after frame
    // 43 holds over the SR copies until the copy expires, then SR takes
    // over. A pause of the judge's tick slack after frame 42's bookkeeping,
    // before its Present (hitch=before_present), puts two clock periods and
    // more between the fixture's clock read and the add-on's stamp of that
    // copy (present_deferred). Until 11-02 the fixture began the hold's
    // window at its own read, so it ended the hold that much before the
    // add-on did and judged the add-on's last held Presents stale_shown, an
    // SR copy readable beside them (frames 49 to 51 at add-on ages 203 to
    // 235 ms, 3 of 3 runs with the restamp disabled); the seed failed so at
    // frame 172 in 1 of 6 fixed-set runs, where the clock ticked between the
    // two reads unprompted.
    {"fg-on-deferred-tag-hitch-before-present-held", "20*sr=local; fg=off,tag=present_list,entry=global; tag=present_list,entry=global; fg=on,tag=present_list,entry=global,untag=after_global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,untag=after_global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,untag=after_global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,sr=tag,dlssg=direct,generated=1; tag=present_list,entry=global,sr=tag,dlssg=direct,state=unprovable,generated=3; tag=present_list,entry=global,sr=tag,dlssg=direct,state=unprovable,generated=3; drs=change,tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable,generated=3; tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable,generated=3; tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable,generated=3; tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable,generated=3; tag=present_list,entry=global,untag=after_global,sr=local,dlssg=direct,state=unprovable; fg=on,drs=change,tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable; fg=on,tag=present_list,entry=global,sr=local,dlssg=direct,state=unprovable,generated=1; tag=present_list,entry=global,sr=local,state=unprovable,generated=1; tag=present_list,entry=global,sr=local,state=unprovable,generated=1; tag=present_list,entry=global,sr=local,state=proven,generated=1; tag=present_list,entry=global,sr=local,generated=1; tag=present_list,entry=global,sr=local,generated=1; tag=present_list,entry=global,sr=local,generated=1,ui=bare,hitch=before_present; tag=present_list,entry=global,untag=after_global,sr=local,generated=1,ui=bare; untag=after_global,sr=other_queue,generated=1,ui=bare,order=second; fg=on,untag=after_global,sr=other_queue,dlssg=compute_unobserved,generated=1,ui=bare,order=second,dlssg_gpu=after_present; sr=local,dlssg=compute_unobserved,generated=1,ui=bare,order=second; sr=local,dlssg=compute_unobserved,generated=1,ui=bare,order=second; sr=local,dlssg=compute_unobserved,ui=bare,order=second; sr=local,dlssg=compute_unobserved,ui=bare; fg=on,sr=local,dlssg=compute_unobserved,ui=bare; sr=local,dlssg=compute_unobserved,order=wait_first; sr=local,dlssg=compute_unobserved,ui=bare,order=wait_first,dlssg_gpu=after_present; fg=on,sr=local,dlssg=compute_unobserved,ui=bare,order=wait_first; sr=local,dlssg=direct_unobserved,ui=bare; tag=now_list,sr=local,ui=bare; 10*sr=local"},
  };

  // Each known gap the timelines found has a witness: an event list whose
  // frames meet its limitation alone, long enough that every run loses depth
  // there. The fixed set runs them after its seeds, and a witness that loses
  // none fails it (the gap closed: its fix removes the row and the witness,
  // which becomes a regression timeline), as the matrix's gap seeds keep a
  // recorded gap's own failing cell. C5's closed on 10-26. Carried: OPEN C's
  // depth twin (found 11-01, an accepted limit since owner ruling 11-02,
  // whose witness keeps running so a change of its behaviour still fails
  // the set): C1-openc's evidence job twinDepth at 14ec980a, its frame 22's
  // SR copy moved to the presenting queue. FG 2x with UI tags, a frame whose
  // FG tag is deferred to its Present (two generated Presents declared, none
  // presented), then a frame with no depth call after a hitch of the source
  // age: depth's window still presents the earlier frame there and holds its
  // copy, past the source age, at the new frame's real Present
  // (continuity:stale at frame 23). With that SR copy on the game's queue (twinDepth's form) it is
  // readable at frame 22's Present in most runs, and the same stale hold
  // follows; in one fixed-set run (REPAIR4's f3-REPAIR4-f1) it completed only
  // after the pause and showed fresh at frame 23 (oracle:unsupplied_fresh),
  // the same lost frame identity, which this row does not excuse.
  struct witness {
    const char *row, *events;
  };

  constexpr std::array<witness, 1> witnesses {{
    {"OPEN-C-depth-unidentified-real", "20*sr=local; fg=on,sr=local,generated=2; tag=evaluate_list,untag=after_global,generated=2,at=inside,ui=bare; tag=present_bare,untag=after_global,sr=local,ui=bare; ui=bare,hitch=between; 5*sr=local,ui=bare"},
  }};

  // The seed of the fixed set's or a soak's index-th timeline.
  std::uint64_t seed_of(std::uint64_t index) {
    generator::seeded mix {0x5e5510e7173e11eeull ^ index};
    return mix.next();
  }

  timeline_case generate(std::uint64_t seed) {
    namespace events = frame_events;
    generator::seeded rng {seed};
    timeline_case out {seed, {}};
    const auto count = min_frames + rng.below(max_frames - min_frames + 1);
    for (unsigned i = 0; i != count; ++i) {
      auto previous = out.frames.empty() ? generator::row {} : out.frames.back();
      previous.resize(events::base_family().dimensions.size());
      out.frames.push_back(generator::sample(events::base_family(), rng, events::drawing(), out.frames.empty() ? nullptr : &previous));
    }
    // The fence events, from a stream of their own.
    generator::seeded fences {seed ^ 0x0f3e5ce0f3e5ce01ull};
    const auto weighted = [](generator::seeded &rng, const unsigned *weights, std::size_t count) {
      auto pick = rng.below(std::accumulate(weights, weights + count, 0u));
      int value = 0;
      while (pick >= weights[value]) {
        pick -= weights[value++];
      }
      return value;
    };
    for (std::size_t i = 0; i != out.frames.size(); ++i) {
      int value = 0;
      if (i && fences.below(1000) < events::order_keep_per_mille) {
        value = out.frames[i - 1][events::order];
      } else {
        value = weighted(fences, events::order_weights, std::size(events::order_weights));
      }
      out.frames[i].push_back(value);
    }
    // The slow intervals, from a stream of their own (until the 10-27 review
    // no seed drew them).
    generator::seeded slow {seed ^ 0x5106e7a1c0ffee05ull};
    for (auto &frame : out.frames) {
      frame.push_back(weighted(slow, events::hitch_weights, std::size(events::hitch_weights)));
      frame.push_back(weighted(slow, events::dlssg_gpu_weights, std::size(events::dlssg_gpu_weights)));
    }
    // The SR work on the game's queue held past its Present, from a stream
    // of its own (10-28).
    generator::seeded sr_slow {seed ^ 0x5e9a7e0ff1a7e5edull};
    for (auto &frame : out.frames) {
      frame.push_back(weighted(sr_slow, events::sr_gpu_weights, std::size(events::sr_gpu_weights)));
      frame.resize(events::family().dimensions.size());
    }
    // Half of the Offs while FG is on come late, after the frame's
    // evaluations and before its Present, from a stream of their own (10-28).
    generator::seeded late {seed ^ 0x1a7e0ff0ff1a7e01ull};
    bool on = false;
    for (auto &frame : out.frames) {
      const auto event = events::of<events::fg_event>(frame, events::fg);
      if (event == events::fg_event::off && on && late.below(2)) {
        frame[events::fg] = int(events::fg_event::off_late);
      }
      on = event == events::fg_event::on ? true : event == events::fg_event::keep ? on : false;
    }
    events::normalize(out.frames);
    return out;
  }

  // The event list: "defaults" or name=value pairs per frame, a run of equal
  // frames as N*frame, frames separated by "; ".
  std::string describe(const std::vector<generator::row> &frames) {
    std::string out;
    for (std::size_t i = 0; i < frames.size();) {
      std::size_t run = 1;
      while (i + run < frames.size() && frames[i + run] == frames[i]) {
        ++run;
      }
      out += out.empty() ? "" : "; ";
      if (run > 1) {
        out += std::to_string(run) + "*";
      }
      out += generator::describe(frame_events::family(), frames[i]);
      i += run;
    }
    return out;
  }

  bool parse_events(const std::string &text, std::vector<generator::row> &out, std::string &error) {
    const auto &owner = frame_events::family();
    out.clear();
    std::size_t start = 0;
    while (start <= text.size()) {
      auto end = text.find(';', start);
      if (end == std::string::npos) {
        end = text.size();
      }
      auto item = text.substr(start, end - start);
      start = end + 1;
      item.erase(0, item.find_first_not_of(' '));
      item.erase(item.find_last_not_of(' ') + 1);
      if (item.empty()) {
        continue;
      }
      unsigned repeat = 1;
      if (const auto star = item.find('*'); star != std::string::npos) {
        repeat = unsigned(std::strtoul(item.c_str(), nullptr, 10));
        item = item.substr(star + 1);
      }
      generator::row value = generator::defaults(owner);
      if (item != "defaults" && !generator::parse(owner, item, value, error)) {
        return false;
      }
      if (!repeat || !owner.legal(value)) {
        error = "the frame '" + item + "' is not one a timeline can carry";
        return false;
      }
      out.insert(out.end(), repeat, value);
    }
    if (out.empty()) {
      error = "the event list holds no frame";
      return false;
    }
    frame_events::normalize(out);
    return true;
  }

  constexpr bool requires_capture(const matrix::outcome &value) {
    return (value.kind == matrix::outcome_kind::capture || value.kind == matrix::outcome_kind::continuous) && !matrix::either(value);
  }

  constexpr bool permits_capture(const matrix::outcome &value) {
    return matrix::permits(value, matrix::outcome_kind::capture) || matrix::permits(value, matrix::outcome_kind::continuous);
  }

  // A refusal the oracle requires alone, of a class a gate decides (the
  // runtime layer's recorded_refusal): fg_scope and no_copy_point name a
  // call that is no copy point.
  constexpr bool recorded_refusal(const matrix::outcome &value) {
    return value.kind == matrix::outcome_kind::refuse && !matrix::either(value) && value.why != matrix::refusal::none && value.why != matrix::refusal::fg_scope && value.why != matrix::refusal::no_copy_point;
  }

  // A recorded class that refuses a decision the oracle requires to capture:
  // a contract class, but not fg_scope, which a live FG scope gives the same
  // viewport's SR input by design (it owns the viewport's depth).
  constexpr bool refuses_capture(refusals::reason_class value) {
    return value >= refusals::reason_class::state_unprovable && value < refusals::reason_class::fg_scope;
  }

  // One failure of a timeline: a continuity violation no row records, or an
  // oracle disagreement. signature is what a shrunk candidate must fail with:
  // the kind and what names its cause (the record's owner and reason, or the
  // decision and its oracle rule).
  struct failure {
    std::string kind;
    unsigned frame {};  // The real frame whose Present (or generated Present) it was judged at.
    std::string text, signature;
  };

  struct run_result {
    std::vector<failure> failures;
    std::array<continuity::row_tally, continuity::gap_count> rows {};
    unsigned presents {}, generated {}, legal {}, shown {}, stalls {}, frames {};
    // Copy decisions judged, required refusals whose gate recorded the
    // oracle's class or that the FG scope decided first, and untags judged
    // by the deferral they must cancel (R7).
    unsigned decisions {}, refusals_matched {}, refusals_preempted {}, cancels {};
    // Reads the add-on recorded as ordered by the queue-order model, each
    // judged by the timeline's own fences (unordered_read).
    unsigned ordered_reads {};
    // Of those, the reads a signal first submitted after them unordered
    // (R1-late-signaller's).
    unsigned late_reads {};
    // Generated Presents of the last UI tag's frame that held its UI
    // decision by depth's window (judge_ui_frame, generated_present).
    unsigned ui_holds {};
    // Presents without frame identity that held the earlier frame's UI
    // decision, left unjudged (ui_unidentified_now).
    unsigned ui_unidentified_holds {};
    std::uint64_t elapsed_ms {};
  };

  struct timeline_fixture: native_provider_fixture {
    using action_t = BOOL (*)(api::effect_runtime *);
    using events = frame_events::dimension;
    action_t recalibrate {};
    // The test add-on's Present of an earlier frame inside the next nominated
    // copy's call, and its withdrawal when no such copy came.
    void (*present_before_next_copy)() {};
    BOOL (*cancel_present_before_next_copy)() {};
    // The test add-on's newest published copy (SunshineStreamlineTestNewestCopy).
    BOOL (*newest_copy_fn)(std::uint64_t *, BOOL *) {};
    // The add-on's totals of depth copies deferred to their frame's Present,
    // of those copied and of the deferrals an untag cancelled.
    void (*deferral_totals)(std::uint64_t *, std::uint64_t *, std::uint64_t *) {};
    // The add-on's lost metadata reads (reads=lost_at_present), its call at
    // the next depth frame (fg_at=depth_read) and the FG mode its last
    // Present used for UI and for depth.
    void (*fail_metadata_reads)(unsigned) {};
    void (*options_at_depth_frame)(void (*)(void *), void *) {};
    BOOL (*present_frame_generation)(api::effect_runtime *, unsigned *, unsigned *, std::uint64_t *) {};
    // The presenting queue's newest depth choice, whose acquisition names
    // the frame depth presents (presenting_floor, the UI's depth window).
    BOOL (*queue_choice)(api::effect_runtime *, sunshine_streamline::depth_capture::depth_choice *) {};
    decltype(&NVSDK_NGX_D3D12_CreateFeature) ngx_create {};
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) ngx_evaluate {};
    const matrix::sl_version version {2, 7, 30};
    sunshine_streamline::abi_v2::viewport view {};
    std::uint32_t next_viewport {5000}, next_scratch {970000};

    // The game's own direct queue, which the presenting queue waits for (as
    // the native provider fixture's sr-other-queue), and DLSS-G's compute
    // queue (as its dlssg-evaluation).
    queue_ring<16> game_work;
    queue_ring<3> dlssg_work;

    // The game's fences that order the presenting queue after each ring's
    // work (frame_events::order_form), created anew for each run and at each
    // FG toggle, and a third queue of the game's that signals one too (a
    // second signaller; it waits for the value first, yet the fence can fall
    // back, signal_again). token: the fence's name in the unordered_read script, one
    // per fence object, so that a fence created at a released one's address
    // is another fence there as it is to the add-on.
    struct order_fence {
      com_ptr<ID3D12Fence> fence;
      std::uint64_t value {};
      const void *token {};
      // A CPU or second signal reached it: the add-on refuses it for good.
      // settled: its first value was seen complete (a wait before is
      // refused).
      bool poisoned {}, settled {};
      // When the presenting queue last waited on it (0 never), and when a
      // frame's copy was last readable at its Present only by relying on it
      // (U-C1: a read in flight that a later signal can conflict, O1).
      std::uint64_t waited {}, relied {};
    };

    order_fence game_order, dlssg_order;
    // The game ring's entries whose recording carried an SR copy the add-on
    // recorded (a list it knows to run on the game's queue).
    std::array<bool, decltype(game_work)::size> game_work_carried {};
    com_ptr<ID3D12CommandQueue> third;
    // The gate DLSS-G's compute queue waits on (dlssg_gpu=after_present):
    // the CPU opens it once the frame's Present returned
    // (fixture_t::after_present), as a GPU-bound game's compute copy is
    // still in flight at its own Present; closed: this frame waits on it.
    com_ptr<ID3D12Fence> dlssg_gate;
    std::uint64_t dlssg_gate_value {};
    const void *dlssg_gate_token {};
    bool dlssg_gate_closed {};
    // The same for the game's own queue's SR work (sr_gpu=after_present).
    com_ptr<ID3D12Fence> sr_gate;
    std::uint64_t sr_gate_value {};
    const void *sr_gate_token {};
    bool sr_gate_closed {};
    // What the game clock's barriers wait for (watch_rings).
    std::array<sunshine_game3d_fixture::game_clock::ring_watch, 4> ring_watches;
    // A late FG Off (fg=off_late) renews the order fences after the frame's
    // Present and its late signals (the frame's work signals the old ones).
    bool renew_after_present {};
    // The frame's late signals, made after its real Present.
    std::vector<order_fence *> late_signals;
    // The game's HUD-less and Backbuffer images: its UI tags and DLSS-G's
    // inputs.
    com_ptr<ID3D12Resource> hudless, backbuffer;
    DXGI_FORMAT color_format {};
    ngx_fixture::parameters dlssg_inputs;
    void *feature {};

    // The run under way.
    std::string where;
    const std::vector<generator::row> *frames {};
    unsigned frame {};
    bool fg_on {};
    unsigned area {};
    sunshine_streamline::abi_v2::frame_token *token {};
    // The depth allocation's latest transition on the game's open list.
    bool evidence_seen {};
    std::uint32_t evidence_state {};
    run_result result;
    // The newest capture a Present showed.
    std::uint64_t last_capture {};
    // When a frame last supplied an API source (legal or permitted), when
    // one last recorded a copy on another queue, and when one last supplied a
    // copy its FG scope accepts.
    std::uint64_t api_supplied {}, other_queue_copy {}, fg_copy {};
    // The fences (script tokens) the frame that last set fg_copy read its
    // DLSS-G copy through, and whether a late signal of one reached it since.
    std::vector<const void *> fg_copy_relied;
    // The capture id of the FG copy fg_copy names (0 unknown).
    std::uint64_t fg_copy_capture {};
    bool fg_copy_lost {};
    // Whether the frame's tag copy is deferred to its Present, and whether
    // it waits there for DLSS-G to evaluate its frame
    // (frame_generation::deferral_evaluations): DLSS-G evaluated a frame
    // since the previous frame whose tag copy was deferred, or none was.
    bool deferred_tag {}, awaits_dlssg {};
    // Whether an add-on evaluation began in the frame being recorded, set
    // where the fixture makes the call that begins it: an SR evaluation, a
    // DLSS-G evaluation, or a tag call the oracle copies or refuses at the
    // call. Not a tag call whose copy the oracle defers to the frame's
    // Present (R2) or holds for the frame's own untag (R7): the add-on
    // nominates nothing there, so the previous frame's window stays open
    // (depth_capture::frame_presenting). Not when a decision is pushed
    // either: decide() and the tag call's decisions come before their calls.
    // A generated Present inside the recording before it presents the last
    // real Present's frame (generated_present, between; owner ruling Q2,
    // 10-30: until then a deferred tag call's decision counted, so after a
    // hitch that frame's own copy shown there was judged stale, soak seeds
    // d12563f8dc22b059 and 2dbc7936fb23f7fa).
    bool evaluation_began {};
    // A frame whose tag copy is deferred to its Present (R2) as a legal
    // source: the add-on's newest published copy right before that Present
    // (newest_copy, never 0 here: 1 stands for none). The copy that Present
    // publishes is the frame's tag copy, whose capture id the judge learns
    // after it (judge::learned_capture); until 10-31 such a copy had none
    // (supply::capture 0), so only the tick bound judged a Present holding an
    // older copy beside it. 0: no such copy.
    std::uint64_t deferred_copy_before {};
    // The oracle's FG scope at the frame's copy decisions, before a late FG
    // Off (fg=off_late) after them: judge_decisions reads a preempting FG
    // rule in that scope (until 10-28 it read the scope after the Off, so
    // seeds that draw a late Off failed a refusal DLSS-G's scope preempted
    // as recorded nowhere outside FG scope).
    bool decided_in_scope {};
    // Whether one of the frame's required copies on another queue is
    // readable at its own Present (execute_elsewhere).
    bool readable_here {};
    // When DLSS-G last evaluated a frame (copied, deferred to a tag or
    // refused), the frames it evaluated, and their count at the previous
    // frame whose tag copy was deferred (deferred_known: one was).
    std::uint64_t dlssg_evaluated {}, dlssg_frames {}, deferred_dlssg_frames {};
    bool deferred_known {};
    // Whether the previous frame supplied an FG copy, so that a live FG
    // scope owns this one (continuity::supply::owned).
    bool fg_supplied {};
    // When FG was last turned on (frame_generation_holds keeps only a copy
    // made since).
    std::uint64_t fg_enabled_at {};
    std::map<std::uint64_t, unsigned> present_frames;

    // What the frame being recorded supplies (the continuity supply log),
    // and its copies recorded on another queue with the completion probe of
    // each the contract copies out (the freshness rule).
    struct frame_supply {
      // immediate: continuity::supply::immediate.
      std::uint8_t legal {}, limited {}, permitted {}, elsewhere {}, immediate {};
      continuity::limitation limit {continuity::limitation::none};
      // made: continuity::supply::made, each source's copy at its deciding
      // call (after any hitch before it); capture: continuity::supply::capture,
      // the copy the add-on published inside that call, if any.
      std::uint64_t tick {};
      std::array<std::uint64_t, 4> made {}, capture {};
      // continuity::supply::completion: each copy's own relied fence.
      std::vector<continuity::supply::completion> completions;
    } supply;

    // The supply of the frame the last real Present presented.
    frame_supply presented_supply;

    // The order fence (its script token, new at each FG toggle) that
    // execute_elsewhere's copy was readable through at its frame's Present
    // before it completed (a relied read), else null.
    const void *relied_on {};

    // A required copy of the frame's last decision, recorded on another
    // queue: the freshness rule's supply, readable once its copy-out
    // completed (every such copy is copied out since U-C1b). Called once its
    // list executed (completion: its ring's). Its made stamp is its deciding
    // call's (offer, before the call, as for every source; until 11-01 a
    // second stamp here at the list's submission was dead: offer had always
    // stamped the decision). A permitted copy may or may not have been made,
    // so it neither counts nor has a completion here.
    // readable: the contract lets the add-on read it at the frame's own
    // Present (execute_elsewhere). Otherwise it stays legal: the frame's
    // Present holds earlier depth (continuity::supply::unordered), and later
    // Presents read it once complete (until 10-27 it was permitted with no
    // completion, so after a CPU or second signal no frame on that queue was
    // judged until FG toggled). A copy readable through a relied read names
    // its order fence (continuity::supply::completion::relied).
    void recorded_elsewhere(std::uint8_t source, std::function<bool()> completion, bool readable) {
      const auto relied = std::exchange(relied_on, nullptr);
      const auto &made = decisions.back();
      if (!made.legal || !requires_capture(made.expected)) {
        return;
      }
      supply.elsewhere |= source;
      readable_here = readable_here || readable;
      if (readable) {
        supply.immediate |= source;
      }
      supply.completions.push_back({source, std::move(completion), relied, false, captured(source)});
    }

    // The add-on's newest published copy: its capture id with
    // continuity::copy_fg_input for an FG scope's input, else 0.
    std::uint64_t newest_copy() const {
      std::uint64_t id {};
      BOOL fg {};
      return newest_copy_fn(&id, &fg) ? id | (fg ? continuity::copy_fg_input : 0) : 0;
    }

    // The copy a call published, read after it: newest_copy() unless it is
    // still before, the one read before the call (the call published none).
    // The first of the frame's copies of source names it (supply::capture).
    void note_copy(std::uint8_t source, std::uint64_t before) {
      const auto after = newest_copy();
      if (after == before) {
        return;
      }
      for (unsigned i = 0; i != supply.capture.size(); ++i) {
        if ((source & (1u << i)) && !supply.capture[i]) {
          supply.capture[i] = after;
        }
      }
    }

    // The capture of the frame's copy of source (0 unknown).
    std::uint64_t captured(std::uint8_t source) const {
      for (unsigned i = 0; i != supply.capture.size(); ++i) {
        if (source & (1u << i)) {
          return supply.capture[i];
        }
      }
      return 0;
    }

    // A refusal recorded since the previous poll, and the step of the frame
    // that recorded it.
    struct polled_refusal {
      const char *step;
      refusals::record value;
    };

    std::vector<polled_refusal> polled;
    std::uint64_t refusal_serial {};

    // One copy decision of the frame and what the oracle names for it.
    struct decision {
      const char *point;
      matrix::outcome expected;
      bool legal {};
      bool dlss_g {};  // DLSS-G's own evaluation: refused by its adapter.
      std::vector<const char *> steps;  // Where its gate decides.
    };

    std::vector<decision> decisions;

    // -- Setup ---------------------------------------------------------------

    void setup(HMODULE sdk) {
      const auto addon = GetModuleHandleW(L"SunshineSBSTest.addon64");
      set_foreground = reinterpret_cast<void (*)(HWND)>(GetProcAddress(addon, "SunshineSbsTestSetForeground"));
      recalibrate = reinterpret_cast<action_t>(GetProcAddress(addon, "SunshineGame3DTestRecalibrate"));
      require(set_foreground && recalibrate, "Test add-on lacks the foreground observer or recalibration");
      present_before_next_copy = reinterpret_cast<void (*)()>(GetProcAddress(addon, "SunshineStreamlineTestPresentBeforeNextCopy"));
      cancel_present_before_next_copy = reinterpret_cast<BOOL (*)()>(GetProcAddress(addon, "SunshineStreamlineTestCancelPresentBeforeNextCopy"));
      require(present_before_next_copy && cancel_present_before_next_copy, "Test add-on lacks its Present-before-copy hook");
      newest_copy_fn = reinterpret_cast<BOOL (*)(std::uint64_t *, BOOL *)>(GetProcAddress(addon, "SunshineStreamlineTestNewestCopy"));
      require(newest_copy_fn, "Test add-on lacks its newest-copy report");
      deferral_totals = reinterpret_cast<decltype(deferral_totals)>(GetProcAddress(addon, "SunshineStreamlineTestDeferrals"));
      require(deferral_totals, "Test add-on lacks its deferral totals");
      fail_metadata_reads = reinterpret_cast<decltype(fail_metadata_reads)>(GetProcAddress(addon, "SunshineStreamlineTestFailMetadataReads"));
      options_at_depth_frame = reinterpret_cast<decltype(options_at_depth_frame)>(GetProcAddress(addon, "SunshineGame3DTestOptionsAtDepthFrame"));
      present_frame_generation = reinterpret_cast<decltype(present_frame_generation)>(GetProcAddress(addon, "SunshineGame3DTestPresentFrameGeneration"));
      require(fail_metadata_reads && options_at_depth_frame && present_frame_generation, "Test add-on lacks its FG mode seams");
      queue_choice = reinterpret_cast<decltype(queue_choice)>(GetProcAddress(addon, "SunshineDepthTestQueueChoice"));
      require(queue_choice, "Test add-on lacks its queue choice report");
      // The export stop at a focus loss would end FG reuse: select the hidden
      // game window in the test observer without moving focus.
      set_foreground(window);
      const auto exe = GetModuleHandleW(nullptr);
      ngx_create = reinterpret_cast<decltype(ngx_create)>(GetProcAddress(exe, "NVSDK_NGX_D3D12_CreateFeature"));
      ngx_evaluate = reinterpret_cast<decltype(ngx_evaluate)>(GetProcAddress(exe, "NVSDK_NGX_D3D12_EvaluateFeature"));
      require(ngx_create && ngx_evaluate, "The timeline runtime lacks its exported NGX entry points");
      const auto getter = bind_streamline(sdk);
      require(getter && sl.new_token && sl.constants && sl.frame_tag && sl.global_tag && sl.evaluate, "The 2.7.30 interposer lacks public SL exports");
      // Discovery is deferred to a Present: only an options function fetched
      // through the hooked getter afterwards is observed.
      require(reshade_log_contains({"Sunshine Streamline: observing 2.x"}, 15000), "Production discovery did not hook the interposer");
      void *function {};
      require(getter(1000, "slDLSSGSetOptions", function) == 0 && function, "The interposer did not expose FG options");
      sl.set_options = reinterpret_cast<set_options_t>(function);
      setup_device();
    }

    // The objects of the game's device the timelines use: its own direct
    // queue, DLSS-G's compute queue, the HUD-less and Backbuffer images and
    // DLSS-G's inputs. The add-on lifecycle fixture creates them again for
    // each game device.
    void setup_device() {
      create_queues();
      auto color = backbuffers[0]->GetDesc();
      color.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      color_format = color.Format;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &color, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(hudless.put())), "Create the game's HUD-less image");
      checked(game->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &color, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(backbuffer.put())), "Create the game's Backbuffer image");
      // The DLSSG.* keys the DLSS-G helpers set (nvsdk_ngx_helpers_dlssg_d3d.h).
      dlssg_inputs.set_resource("DLSSG.Depth", selected->resource.p);
      dlssg_inputs.set_resource("DLSSG.HUDLess", hudless.p);
      dlssg_inputs.set_resource("DLSSG.Backbuffer", backbuffer.p);
      dlssg_inputs.set_ui("DLSSG.DepthInverted", 1);
      dlssg_inputs.set_ui("DLSSG.MultiFrameIndex", 1);
      dlssg_inputs.set_ui("DLSSG.MultiFrameCount", 1);
      watch_rings();
      // The native observer installs its hooks for the new queues before any
      // timeline is judged.
      render_tracked_depth = {};
      for (const auto until = clock.now() + sunshine_freshness::association_ms; clock.now() < until;) {
        present_frame();
      }
    }

    // The game's queues each barrier of the game clock waits for, so that
    // their work completed before game time moves (test_game3d_clock.h): the
    // game's ring and DLSS-G's ring (their done fences), and each ring's
    // order fence up to the ring's own value (order_fence::value). That bound
    // is from below only. A value the CPU or the third queue signals ahead
    // (order_form cpu and second) is not waited for: the ring's own signal
    // of the value before it can land after that one, so the fence falls
    // back to the ring's value (it does in real time too: the third queue's
    // wait for that value is met by its own earlier signal one ahead). So a
    // barrier can also pass on the value ahead before the ring's own signal
    // landed, and that fence can still fall back after game time moved; the
    // add-on poisons such a fence (second signaller), so no decision relies
    // on its value. Inside a frame whose gate holds a ring's work until after
    // its Present (sr_gpu or dlssg_gpu after_present) that ring and its order
    // fence are not waited for: the gate opens after the Present, and the
    // frame's boundary waits then.
    void watch_rings() {
      const auto sr_held = [this] {
        return sr_gate_closed;
      };
      const auto dlssg_held = [this] {
        return dlssg_gate_closed;
      };
      ring_watches = {clock.watch("game_work", game_work, sr_held), clock.watch("game_order", &game_order.fence.p, &game_order.value, sr_held), clock.watch("dlssg_work", dlssg_work, dlssg_held), clock.watch("dlssg_order", &dlssg_order.fence.p, &dlssg_order.value, dlssg_held)};
    }

    void create_queues() {
      game_work.create(game.p, D3D12_COMMAND_LIST_TYPE_DIRECT);
      dlssg_work.create(game.p, D3D12_COMMAND_LIST_TYPE_COMPUTE);
      D3D12_COMMAND_QUEUE_DESC desc {};
      desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(game->CreateCommandQueue(&desc, IID_PPV_ARGS(third.put())), "Create the game's third queue");
      checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(dlssg_gate.put())), "Create DLSS-G's compute gate");
      dlssg_gate_value = 0;
      dlssg_gate_token = script_fence_token();
      dlssg_gate_closed = false;
      checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(sr_gate.put())), "Create the game queue's SR gate");
      sr_gate_value = 0;
      sr_gate_token = script_fence_token();
      sr_gate_closed = false;
      renew_order();
    }

    // Opens DLSS-G's compute gate if this frame closed it.
    void open_dlssg_gate() {
      if (!std::exchange(dlssg_gate_closed, false)) {
        return;
      }
      checked(dlssg_gate->Signal(dlssg_gate_value), "Open DLSS-G's compute gate after the frame's Present");
      script_order.cpu_signal(dlssg_gate_token, dlssg_gate_value);
    }

    // Opens the game queue's SR gate if this frame closed it.
    void open_sr_gate() {
      if (!std::exchange(sr_gate_closed, false)) {
        return;
      }
      checked(sr_gate->Signal(sr_gate_value), "Open the game queue's SR gate after the frame's Present");
      script_order.cpu_signal(sr_gate_token, sr_gate_value);
    }

    // A fence's waits and signals complete before it goes, as a game's.
    void finish_order(order_fence &order) {
      if (order.fence.p) {
        checked(order.fence->SetEventOnCompletion(order.value, nullptr), "Wait for an order fence's last value");
      }
      order.fence.reset();
    }

    // Both order fences created anew: a run's start, or an FG toggle.
    void renew_order() {
      for (auto *order : {&game_order, &dlssg_order}) {
        finish_order(*order);
        checked(game->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(order->fence.put())), "Create an order fence");
        order->value = order->waited = order->relied = 0;
        order->token = script_fence_token();
        order->poisoned = order->settled = false;
      }
    }

    // The add-on lifecycle fixture ends the game's device.
    void release_order() {
      open_dlssg_gate();
      open_sr_gate();
      finish_order(game_order);
      finish_order(dlssg_order);
      third.reset();
      dlssg_gate.reset();
      sr_gate.reset();
    }

    // The ring's entry n executed on its queue, which then signals its order
    // fence; the presenting queue (or the CPU) waits for that value as the
    // frame's fence events say (frame_events::order_form), and the CPU or
    // the third queue signals it again, now (cpu, second) or after the
    // frame's Present (late). Whether the contract lets the add-on read the
    // frame's copy at its Present (docs/reshade-sbs.md, Queue order): it is
    // complete (the CPU waited for it), or the presenting queue waits for a
    // fence the add-on may rely on, one that no CPU or second signal reached
    // (sticky poison, O3) and that settled: its first value was seen
    // complete before this frame's calls on it (a wait before that is
    // refused for good).
    // mutable_copy: the frame's copy on the ring is not immutable for the
    // presenting queue (a cold list, below): neither the order fence nor a
    // CPU wait makes it readable at its Present.
    template<unsigned N>
    bool execute_elsewhere(queue_ring<N> &ring, unsigned n, order_fence &order, bool mutable_copy = false) {
      using frame_events::order_form;
      const auto form = frame_events::of<order_form>((*frames)[frame], events::order);
      const auto next = order.value + 1;
      if (!order.settled && order.value && order.fence->GetCompletedValue() >= 1) {
        order.settled = true;
      }
      const bool relied = order.settled && !order.poisoned && !mutable_copy;
      const auto wait = [&] {
        checked(queue->Wait(order.fence.p, next), "Order the presenting queue after a ring's queue");
        script_order.wait(queue.p, order.token, next);
        order.waited = clock.now();
      };
      if (form == order_form::wait_first) {
        wait();
      }
      ring.execute(n);
      checked(ring.queue->Signal(order.fence.p, next), "Signal a ring's order fence");
      script_order.signal(ring.queue.p, order.token, next);
      order.value = next;
      if (form == order_form::cpu_wait || form == order_form::cpu) {
        // On the ring's own fence, which nobody else signals.
        checked(ring.done->SetEventOnCompletion(ring.value, nullptr), "Wait for a ring's work on the CPU");
      } else if (form != order_form::wait_first) {
        wait();
      }
      if (form == order_form::cpu || form == order_form::second) {
        signal_again(order, form == order_form::cpu, true);
      } else if (form == order_form::late) {
        late_signals.push_back(&order);
      }
      const bool complete = form == order_form::cpu_wait || form == order_form::cpu;
      relied_on = nullptr;
      if (!complete && relied && form != order_form::second) {
        order.relied = clock.now();
        relied_on = order.token;
      }
      return !mutable_copy && (complete || (relied && form != order_form::second));
    }

    // A signal of the order fence by the CPU, or by the third queue once its
    // last value was reached: of that value
    // (late), or one ahead of it (cpu, second), which the ring's next signal
    // repeats, so the presenting queue's next wait is reached before that
    // ring's work: an order the add-on must refuse (sticky poison, O3),
    // which unordered_read judges. The first such signal of a fence may
    // conflict with a read that relied on it (O1): the copies read through
    // it are no longer holdable from here (judge::late_signal). Until 10-27
    // a late window then tagged every legal source of every real frame
    // within the source age with R1-late-signaller, which excused any loss
    // there, also of a copy the presenting queue could read.
    // The fence can fall back (seen 11-02, also in real time): one ahead,
    // after the ring's work but before the ring's own signal of the last
    // value (the CPU waits for the ring's done fence, which the ring
    // signals first; in consecutive second frames the third queue's wait is
    // met by its own earlier signal one ahead), that signal lands later and
    // sets the fence back to the ring's value until the ring's next signal.
    void signal_again(order_fence &order, bool cpu, bool ahead) {
      if (!order.poisoned) {
        continuity_log.judge.late_signal(order.token);
        // The scope's held FG copy was read through it: the add-on stops
        // holding it (0a16e59a), so it keeps the viewport's SR out no more.
        fg_copy_lost = fg_copy_lost || std::find(fg_copy_relied.begin(), fg_copy_relied.end(), order.token) != fg_copy_relied.end();
      }
      order.poisoned = true;
      const auto value = order.value + (ahead ? 1 : 0);
      if (cpu) {
        checked(order.fence->Signal(value), "Signal an order fence from the CPU");
        script_order.cpu_signal(order.token, value);
        return;
      }
      checked(third->Wait(order.fence.p, order.value), "Order the third queue after a ring's signal");
      script_order.wait(third.p, order.token, order.value);
      checked(third->Signal(order.fence.p, value), "Signal an order fence from the third queue");
      script_order.signal(third.p, order.token, value);
    }

    // Whether a first signal of order now may conflict a read in flight.
    static bool late_relied(const order_fence &order, std::uint64_t now) {
      return !order.poisoned && order.relied && now - order.relied < sunshine_freshness::source_age_ms + continuity::tag_slack_ms;
    }

    // Retires every ring recording (as a game does at its end), once their
    // copies are older than the source age.
    void drain() {
      open_dlssg_gate();
      open_sr_gate();
      game_work.drain();
      dlssg_work.drain();
    }

    // -- The oracle's view of the frame's decisions --------------------------

    frame_events::state_form state_choice() const {
      return frame_events::of<frame_events::state_form>((*frames)[frame], events::state);
    }

    // A depth input's declaration under the frame's state choice: the state
    // at the use its lifecycle names (SLT:357), as the frame keeps it there:
    // at its call (OnlyValidNow, and an evaluation's own input at that
    // evaluation), at its frame's Present (ValidUntilPresent: every frame
    // ends in UNORDERED_ACCESS) or at its evaluate (ValidUntilEvaluate: a
    // proven frame stays in NON_PIXEL_SHADER_RESOURCE until its end); none
    // (UINT_MAX) for an unprovable choice. A wrong declaration of a later use
    // is a game's bug the oracle cannot describe at a deferred copy.
    std::uint32_t declared_state(matrix::lifecycle life) const {
      using frame_events::state_form;
      if (state_choice() == state_form::unprovable) {
        return UINT32_MAX;
      }
      switch (life) {
        case matrix::lifecycle::valid_until_present:
          return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case matrix::lifecycle::valid_until_evaluate:
          return state_choice() == state_form::proven ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        default:
          return selected->state;
      }
    }

    // The depth allocation as the open list records it.
    matrix::resource_state recorded(bool seen, std::uint32_t state) const {
      matrix::resource_state out {D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
      if (seen) {
        out.seen = matrix::evidence::transition;
        out.observed = state;
      }
      return out;
    }

    // A transition of the depth allocation on the game's open list.
    void move(D3D12_RESOURCE_STATES to) {
      if (selected->state == to) {
        return;
      }
      transition(commands.p, selected->resource.p, selected->state, to);
      selected->state = to;
      evidence_seen = true;
      evidence_state = to;
    }

    bool frame_generation_scope() const {
      matrix::sl_case value;
      value.version = version;
      value.fg = fg_on ? matrix::fg_mode::on : matrix::fg_mode::off;
      return matrix::detail::frame_generation_scope(value);
    }

    // The frame's depth tag at a copy point, in the oracle's terms.
    matrix::sl_case tag_case(matrix::copy_point point, std::uint32_t generated_before) const {
      using namespace frame_events;
      const auto &value = (*frames)[frame];
      const auto form = unsigned(of<tag_form>(value, events::tag)) - 1;
      matrix::sl_case out;
      out.version = version;
      out.entry = of<entry_form>(value, events::entry) == entry_form::frame ? matrix::tag_entry::set_tag_for_frame : matrix::tag_entry::set_tag;
      out.life = matrix::lifecycle(form % 3);
      out.commands = form >= 3 ? matrix::tag_commands::none : matrix::tag_commands::game_list;
      out.at = point;
      out.generated_presents = point == matrix::copy_point::tag_call ? 0 : generated_before;
      out.extent = matrix::extent_form::padded;
      out.fg = fg_on ? matrix::fg_mode::on : matrix::fg_mode::off;
      const auto untag = of<untag_form>(value, events::untag);
      out.untag = untag == untag_form::after_frame || untag == untag_form::after_global ? matrix::untag_pattern::null_after_each : matrix::untag_pattern::none;
      out.declared_state = declared_state(out.life);
      out.depth = recorded(evidence_seen, evidence_state);
      return out;
    }

    // R7: the frame's own untag cancels the copy its tag deferred to the
    // frame's Present (the oracle's S9 HOLD of a tag whose outcome without
    // the untag is that deferral).
    static bool cancels_deferral(const matrix::sl_case &value, const matrix::outcome &expected) {
      if (value.untag != matrix::untag_pattern::null_after_each || expected.kind != matrix::outcome_kind::hold || matrix::either(expected)) {
        return false;
      }
      auto kept = value;
      kept.untag = matrix::untag_pattern::none;
      const auto deferral = matrix::expect(kept);
      return deferral.deferred && permits_capture(deferral);
    }

    std::uint64_t deferral_total(unsigned which) const {
      std::uint64_t value[3] {};
      deferral_totals(&value[0], &value[1], &value[2]);
      return value[which];
    }

    // The oracle's outcome for the frame's tag at a copy point. The oracle
    // names an untag through the tag's own entry (R7, untag_pattern); a
    // token-less untag of a frame tag (slSetTag after slSetTagForFrame) is
    // outside it, so that pairing is EITHER: the outcome with the untag or
    // without it.
    matrix::outcome tag_outcome(const matrix::sl_case &value) const {
      const auto untagged = matrix::expect(value);
      if (value.entry != matrix::tag_entry::set_tag_for_frame || frame_events::of<frame_events::untag_form>((*frames)[frame], events::untag) != frame_events::untag_form::after_global) {
        return untagged;
      }
      auto kept = value;
      kept.untag = matrix::untag_pattern::none;
      const auto copied = matrix::expect(kept);
      return copied.kind == untagged.kind ? untagged : matrix::detail::either(copied, untagged);
    }

    // An SR evaluation's own depth input (lifecycle 1 on the evaluation's
    // list), from what that list records, with the generated Presents that
    // land inside that evaluation.
    matrix::sl_case local_case(std::uint32_t declared, bool seen, std::uint32_t state, std::uint32_t generated = 0) const {
      matrix::sl_case out;
      out.version = version;
      out.entry = matrix::tag_entry::local_array;
      out.at = matrix::copy_point::same_frame_evaluate;
      out.generated_presents = generated;
      out.extent = matrix::extent_form::padded;
      out.fg = fg_on ? matrix::fg_mode::on : matrix::fg_mode::off;
      out.declared_state = declared;
      out.depth = recorded(seen, state);
      return out;
    }

    template<class Case>
    void decide(const char *point, const Case &value, std::uint8_t source, std::initializer_list<const char *> steps, bool dlss_g = false, continuity::limitation meets = continuity::limitation::none) {
      decision made {point, matrix::expect(value), matrix::legal(value), dlss_g, steps};
      offer(made, source, meets);
      decisions.push_back(made);
    }

    // The frame's last decision, if it requires a capture, copies at its
    // call on the game's list: readable at the frame's own Present.
    void immediate(std::uint8_t source) {
      const auto &made = decisions.back();
      if (made.legal && requires_capture(made.expected)) {
        supply.immediate |= source;
      }
    }

    // A decision's supply to the continuity judge.
    void offer(const decision &value, std::uint8_t source, continuity::limitation meets = continuity::limitation::none) {
      if (!value.legal) {
        return;
      }
      if (requires_capture(value.expected)) {
        if (meets == continuity::limitation::none) {
          supply.legal |= source;
        } else {
          supply.limited |= source;
          supply.limit = supply.limit == continuity::limitation::none ? meets : supply.limit;
        }
      } else if (permits_capture(value.expected)) {
        supply.permitted |= source;
      } else {
        return;
      }
      const auto now = clock.now();
      supply.tick = supply.tick ? supply.tick : now;
      for (unsigned i = 0; i != supply.made.size(); ++i) {
        if ((source & (1u << i)) && !supply.made[i]) {
          supply.made[i] = now;
        }
      }
    }

    // -- Refusals ------------------------------------------------------------

    void poll(const char *step) {
      const auto value = last_refusal();
      if (value.serial > refusal_serial) {
        refusal_serial = value.serial;
        polled.push_back({step, value});
      }
    }

    bool mine(const refusals::record &value, bool dlss_g) const {
      return dlss_g ? value.provider == refusals::adapter::dlss_g : value.provider == refusals::adapter::streamline && value.viewport == view.value;
    }

    void fail(const char *kind, const std::string &text, unsigned at, const std::string &cause) {
      result.failures.push_back({kind, at, text, std::string(kind) + "/" + cause});
    }

    std::string frame_text(unsigned at) const {
      return "frame " + std::to_string(at) + " {" + generator::describe(frame_events::family(), (*frames)[at]) + "}";
    }

    // Each decision of the frame against the refusals its gates recorded.
    void judge_decisions() {
      for (const auto &made : decisions) {
        if (!made.legal) {
          continue;
        }
        // The refusals its gates recorded, oldest first (a poll sees the
        // newest of each step).
        std::vector<const polled_refusal *> seen;
        for (const auto &value : polled) {
          if (mine(value.value, made.dlss_g) && std::find_if(made.steps.begin(), made.steps.end(), [&](const char *step) {
                                                  return !std::strcmp(step, value.step);
                                                }) != made.steps.end()) {
            seen.push_back(&value);
          }
        }
        const auto describe_seen = [&] {
          std::string out;
          for (const auto *value : seen) {
            char text[160];
            std::snprintf(text, sizeof(text), "%s%s at %s (class %u, gate %u, stage %u)", out.empty() ? "" : "; ", value->value.name, value->step, unsigned(value->value.reason), unsigned(value->value.where), unsigned(value->value.stage));
            out += text;
          }
          if (!out.empty()) {
            return out;
          }
          // The frame's other refusals, which name the gate that decided.
          out = "none (the frame's:";
          for (const auto &value : polled) {
            char text[200];
            std::snprintf(text, sizeof(text), " %s at %s adapter %u viewport %u class %u gate %u stage %u;", value.value.name, value.step, unsigned(value.value.provider), value.value.viewport, unsigned(value.value.reason), unsigned(value.value.where), unsigned(value.value.stage));
            out += text;
          }
          return out + ")";
        };
        const auto rule = made.expected.by ? made.expected.by->id : "?";
        const auto contract = std::any_of(seen.begin(), seen.end(), [](const polled_refusal *value) {
          return refuses_capture(value->value.reason);
        });
        ++result.decisions;
        if (recorded_refusal(made.expected)) {
          // In a session the FG scope may decide the call first (it owns its
          // viewport's SR inputs, the same frame's untag of its tag, DLSS-G's
          // evaluation defers to its tag copies): its gate then records
          // nothing, or a class outside the contract's (fg_scope, hold), the
          // preempting rule's evidence. Outside the oracle's FG scope no rule
          // preempts a gate, so a required refusal recorded nowhere fails (a
          // copy the contract refuses may have been made). A contract class
          // it records is the oracle's.
          const auto why = refusals::reason_class(unsigned(made.expected.why));
          const bool matched = std::any_of(seen.begin(), seen.end(), [&](const polled_refusal *value) {
            return value->value.reason == why;
          });
          const bool preempted = !matched && !contract && (!seen.empty() || decided_in_scope);
          result.refusals_matched += matched;
          result.refusals_preempted += preempted;
          if (!matched && !contract && !preempted) {
            fail("oracle:refusal_unrecorded", frame_text(frame) + " " + made.point + ": the oracle requires a refusal of class " + std::to_string(unsigned(made.expected.why)) + " (" + rule + ") outside FG scope, recorded: " + describe_seen(), frame, std::string(made.point) + "/" + rule);
          }
          if (contract && !matched) {
            fail("oracle:refusal_class", frame_text(frame) + " " + made.point + ": the oracle requires a refusal of class " + std::to_string(unsigned(made.expected.why)) + " (" + rule + "), recorded: " + describe_seen(), frame, std::string(made.point) + "/" + rule);
          }
        } else if (requires_capture(made.expected) && std::any_of(seen.begin(), seen.end(), [](const polled_refusal *value) {
                     return refuses_capture(value->value.reason);
                   })) {
          fail("oracle:capture_refused", frame_text(frame) + " " + made.point + ": the oracle requires a capture (" + rule + "), recorded refusal: " + describe_seen(), frame, std::string(made.point) + "/" + rule);
        }
      }
    }

    // -- One frame -----------------------------------------------------------

    sunshine_streamline::abi_v2::resource depth_input(std::uint32_t declared) const {
      sunshine_streamline::abi_v2::resource out {};
      out.base = {nullptr, sunshine_streamline::resource_guid, 1};
      out.type = texture_type();
      out.native = selected->resource.p;
      out.state = declared;
      out.width = selected->width;
      out.height = selected->height;
      out.native_format = DXGI_FORMAT_R32_FLOAT;
      out.mip_levels = out.array_layers = 1;
      return out;
    }

    sunshine_streamline::abi_v2::resource color_input(ID3D12Resource *native) const {
      sunshine_streamline::abi_v2::resource out {};
      out.base = {nullptr, sunshine_streamline::resource_guid, 1};
      out.type = texture_type();
      out.native = native;
      out.state = unsigned(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      out.width = width;
      out.height = height;
      out.native_format = unsigned(color_format);
      out.mip_levels = out.array_layers = 1;
      return out;
    }

    void returned(std::int32_t value, const char *call) {
      require(value == 0, (std::string("The add-on changed the SDK result of ") + call).c_str());
    }

    // A generated Present of an earlier frame while the game records this
    // one, or after this one's Present: it supplies nothing, and may show
    // depth only as the capture the last real Present showed.
    void generated_present(bool inside) {
      continuity_log.generated = true;
      // It presents the last real Present's frame as that Present did after
      // that Present (between), or inside this frame's recording before the
      // add-on began an evaluation of this frame (evaluation_began: a tag
      // call it defers to this frame's Present begins none, so it still
      // presents that frame, frame_presenting).
      continuity_log.between = !inside || !evaluation_began;

      struct end_generated {
        ~end_generated() {
          continuity_log.generated = continuity_log.between = false;
        }
      } scope;

      lose_reads(true);
      if (inside) {
        checked(swapchain->Present(0, 0), "Generated Present inside the recording");
      } else {
        render_tracked_depth = {};
        present_frame();
      }
      lose_reads(false);
      ++result.generated;
      judge_present_modes(inside ? "a generated Present inside its recording" : "a generated Present after it");
      generated_since_real += continuity_log.between;
      ui_window_presents += continuity_log.between;
      // Still no frame identity (ui_unidentified): a generated Present
      // between, within the earlier frame's count.
      ui_unidentified_now = ui_unidentified && continuity_log.between && ui_window_presents <= declared_generated;
      judge_ui_frame(inside ? "a generated Present inside its recording" : "a generated Present after it",
        continuity_log.between && inside ? frame - 1 : frame, continuity_log.between ? generated_since_real : 1);
      ui_unidentified_now = false;
      poll("generated");
      if (!record_history.records.empty()) {
        present_frames[record_history.records.back().present] = frame;
      }
      judge_shown(inside ? "a generated Present inside its recording" : "a generated Present after it");
    }

    // A Present shows a fresh copy (a capture no Present showed before) only
    // while a frame within the source age supplied an API source the oracle
    // allows, or a copy recorded on another queue may still become readable:
    // a refused or held decision never shows new depth. A deferred copy
    // (R2) is read at the Present after the one it precedes, and a copy its
    // FG scope kept out is shown once the scope falls silent, so a fresh
    // capture belongs to some frame of that window, not necessarily the
    // Present's own. A fresh copy of an FG scope's own input needs an FG
    // source in that window (an FG tag or DLSS-G's input the oracle allows):
    // the SR inputs of the frames an FG tag's held or refused decision left
    // (S9 and R7, an untag that cancels the deferred copy) supply none. The
    // frame whose supply the fixture holds (the one being recorded, or the
    // one whose real and generated Presents these are) supplies its Presents
    // whatever a hitch since its calls (until the 10-27 review only the
    // source age of the clock counted, so a frame's own FG copy shown at its
    // own Present after a hitch failed as unsupplied: the review's F5).
    void judge_shown(const char *which) {
      const auto s = inspect().source;
      if (trace) {
        trace_present(which, s);
      }
      if (!(s.selected && s.ready)) {
        return;
      }
      const bool fresh = !s.reused_depth && s.current.capture && s.current.capture != last_capture;
      const auto now = clock.now();
      const auto within = [now](std::uint64_t tick, std::uint64_t slack) {
        return tick && now - tick < sunshine_freshness::source_age_ms + slack;
      };
      // A generated Present that presents the last real Present's frame
      // (continuity_log.between) is supplied by that frame.
      const auto &frame_supply = continuity_log.between ? presented_supply : supply;
      const auto own = std::uint8_t(frame_supply.legal | frame_supply.limited | frame_supply.permitted);
      const bool own_api = own & continuity::api_sources;
      const bool own_fg = fg_on && frame_generation_scope() && (own & (continuity::tag_copy | continuity::dlssg_input));
      if (fresh && !other_queue_pending()) {
        if (!own_api && !within(api_supplied, continuity::tick_slack_ms)) {
          fail("oracle:unsupplied_fresh", frame_text(frame) + ": " + which + " showed a fresh copy (capture " + std::to_string(s.current.capture) + ") with no API source supplied within the source age", frame, s.frame_generation_input ? "fg_input" : "scene");
        } else if (s.frame_generation_input && !own_fg && !within(fg_copy, continuity::tag_slack_ms)) {
          fail("oracle:unsupplied_fresh", frame_text(frame) + ": " + which + " showed a fresh copy of an FG scope's input (capture " + std::to_string(s.current.capture) + ") with no FG source supplied within the source age", frame, "fg_source");
        }
      }
      last_capture = s.current.capture;
    }

    // --trace: each Present's frame record, the supply it was judged with
    // and the provider's source, for reading a reproducer.
    bool trace {};

    void trace_present(const char *which, const sunshine_streamline::provider::source_status &s) {
      if (record_history.records.empty()) {
        return;
      }
      const auto &r = record_history.records.back();
      const auto owner = r.depth_owner == sunshine_game3d::source_slot::none ? sunshine_game3d::source_slot::hold : r.depth_owner;
      std::printf("TRACE frame=%u %s present=%llu tick=%llu owner=%s state=%s reason=%s fg=%s/%s scene=%s/%s legal=0x%x limited=0x%x permitted=0x%x shown=%u capture=%llu reused=%u fg_input=%u copies=%llu/%llu/%llu record=%llu\n", frame, which, static_cast<unsigned long long>(r.present), static_cast<unsigned long long>(r.tick_ms), sunshine_game3d::name(r.depth_owner), sunshine_game3d::name(r[owner].state), r[owner].reason, sunshine_game3d::name(r[sunshine_game3d::source_slot::fg_depth].state), r[sunshine_game3d::source_slot::fg_depth].reason, sunshine_game3d::name(r[sunshine_game3d::source_slot::scene_depth].state), r[sunshine_game3d::source_slot::scene_depth].reason, unsigned(supply.legal), unsigned(supply.limited), unsigned(supply.permitted), unsigned(s.selected && s.ready), static_cast<unsigned long long>(s.current.capture), unsigned(s.reused_depth), unsigned(s.frame_generation_input), static_cast<unsigned long long>(supply.capture[0]), static_cast<unsigned long long>(supply.capture[1] & ~continuity::copy_fg_input), static_cast<unsigned long long>(supply.capture[2] & ~continuity::copy_fg_input), static_cast<unsigned long long>(r.depth_capture));
    }

    bool other_queue_pending() const {
      return other_queue_copy && clock.now() - other_queue_copy < sunshine_freshness::source_age_ms + continuity::tag_slack_ms;
    }

    // One real frame's calls on the game's open list: the decoy and the
    // tracked depth drawn, then the frame's events in the order a game
    // makes them.
    void frame_body() {
      using namespace frame_events;
      using namespace sunshine_streamline;
      const auto &value = (*frames)[frame];
      const auto command = [this] {
        return reinterpret_cast<void *>(game_native_command);
      };
      evidence_seen = false;
      deferred_tag = awaits_dlssg = readable_here = evaluation_began = false;
      // A hitch between frames: after the previous frame's Presents, before
      // this frame's first call and its stamp.
      const auto hitch = of<hitch_form>(value, events::hitch);
      if (hitch == hitch_form::between) {
        clock.hitch(hitch_ms(hitch));
      }
      draw(*decoy);
      supply.permitted |= continuity::generic_boundary;
      supply.tick = clock.now();
      supply.made[3] = supply.tick;
      // A hitch before the frame's calls, after its stamp.
      if (hitch == hitch_form::brief || hitch == hitch_form::before_sources) {
        clock.hitch(hitch_ms(hitch));
      }
      require(selected->state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "A timeline frame began with its depth outside UNORDERED_ACCESS");
      write_depth_pixels();
      const auto choice = state_choice();
      if (choice == state_form::proven) {
        move(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      } else if (choice == state_form::recorded_other) {
        move(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
      const auto index = ++fg_frame;
      require(sl.new_token(token, &index) == 0 && token, "The interposer did not supply a frame token");
      returned(sl.constants(camera_constants(false), *token, view), "slSetConstants");
      const abi_v2::resource_tag untag_depth {{nullptr, tag_guid, 1}, nullptr, 0, 1, {}};
      const auto untag = of<untag_form>(value, events::untag);
      if (untag == untag_form::before_global) {
        returned(sl.global_tag(view, &untag_depth, 1, nullptr), "the untag before the frame's tag");
        poll("untag");
      }
      const auto form = of<tag_form>(value, events::tag);
      const auto sr_choice = of<sr_form>(value, events::sr);
      const auto placement = of<generated_at>(value, events::at);
      const auto generated_inside = placement == generated_at::inside ? std::uint32_t(value[events::generated]) : 0u;
      const auto generated_in_evaluate = placement == generated_at::evaluate ? std::uint32_t(value[events::generated]) : 0u;
      void *tag_commands {};
      // R7: the oracle holds the tag's deferred copy for the frame's untag,
      // and the add-on deferred it at the tag call.
      bool cancel_expected = false;
      if (form != tag_form::none) {
        const auto tag_call = tag_case(matrix::copy_point::tag_call, 0);
        const bool frame_entry = tag_call.entry == matrix::tag_entry::set_tag_for_frame;
        tag_commands = tag_call.commands == matrix::tag_commands::game_list ? command() : nullptr;
        // Its copy points: the tag call (or the copy before the frame's
        // Present it defers, R2), and an SR evaluation of it. The frame's
        // Present shows whichever copied (the runtime layer's at_present):
        // the earliest point whose outcome requires a capture.
        decision call {"tag_call", tag_outcome(tag_call), matrix::legal(tag_call), false, {"tag_call", "untag", "ui", "generated", "present"}};
        decisions.push_back(call);
        cancel_expected = call.legal && cancels_deferral(tag_call, call.expected);
        deferred_tag = call.legal && permits_capture(call.expected) && call.expected.deferred;
        // The call begins an evaluation where the oracle copies or refuses it
        // at the call, not where it defers its copy to the frame's Present
        // (R2) or holds it for the frame's untag (R7).
        const bool begins_evaluation = !(deferred_tag || cancel_expected);
        if (deferred_tag) {
          awaits_dlssg = !deferred_known || deferred_dlssg_frames != dlssg_frames;
          deferred_known = true;
          deferred_dlssg_frames = dlssg_frames;
        }
        bool supplied = false;
        if (call.legal && requires_capture(call.expected)) {
          offer(call, continuity::tag_copy);
          supplied = true;
          // Copied at its call on the game's list (not deferred to the
          // frame's Present, R2): readable at its own Present. So is a copy
          // deferred to the frame's Present (R2) that no untag cancels: the
          // add-on records it on the presenting queue's immediate list right
          // before that Present decides, however long the frame took after
          // its tag call (until 10-28 the Present-to-Present rule took a
          // hitch there for a stall's warm-up, which passed the add-on's mono
          // Present after it: open item A, and the fixed set's seed
          // c28965dca8d7c76c at frame 64, three flat Presents).
          if ((!call.expected.deferred && tag_commands) || (call.expected.deferred && !cancel_expected)) {
            supply.immediate |= continuity::tag_copy;
          }
        }
        if (sr_choice == sr_form::tag) {
          const auto evaluated = tag_case(matrix::copy_point::same_frame_evaluate, generated_inside);
          decision at_evaluate {"tag_at_evaluate", tag_outcome(evaluated), matrix::legal(evaluated), false, {"evaluate"}};
          decisions.push_back(at_evaluate);
          if (!supplied && at_evaluate.legal && requires_capture(at_evaluate.expected)) {
            offer(at_evaluate, continuity::sr_evaluate);
            supply.immediate |= continuity::sr_evaluate;
            supplied = true;
          } else if (!supplied && at_evaluate.legal && permits_capture(at_evaluate.expected)) {
            offer(at_evaluate, continuity::sr_evaluate);
          }
        }
        if (!supplied && call.legal && permits_capture(call.expected)) {
          offer(call, continuity::tag_copy);
        }
        const auto resource = depth_input(tag_call.declared_state);
        const abi_v2::resource_tag depth_tag {{nullptr, tag_guid, 1}, &resource, 0, std::uint32_t(tag_call.life), active_area};
        const auto deferred_before = deferral_total(0);
        const auto copy_before = newest_copy();
        evaluation_began = evaluation_began || begins_evaluation;
        returned(frame_entry ? sl.frame_tag(*token, view, &depth_tag, 1, tag_commands) : sl.global_tag(view, &depth_tag, 1, tag_commands), "the depth tag call");
        note_copy(continuity::tag_copy, copy_before);
        cancel_expected = cancel_expected && deferral_total(0) != deferred_before;
        poll("tag_call");
      }
      if (choice == state_form::recorded_other) {
        move(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
      const auto cancelled_before = deferral_total(2);
      if (untag == untag_form::after_frame) {
        returned(sl.frame_tag(*token, view, &untag_depth, 1, tag_commands), "the frame's token untag");
        poll("untag");
      } else if (untag == untag_form::after_global) {
        returned(sl.global_tag(view, &untag_depth, 1, tag_commands), "the untag after the tag");
        poll("untag");
      }
      result.cancels += cancel_expected;
      if (cancel_expected && deferral_total(2) == cancelled_before) {
        fail("oracle:deferral_kept", frame_text(frame) + " untag: the oracle holds the tag's copy deferred to the frame's Present for the frame's untag (R7, S9), and the untag cancelled no deferral the tag call made", frame, "untag/S9");
      }
      const auto ui_choice = of<ui_form>(value, events::ui);
      if (ui_choice != ui_form::none) {
        const auto hudless_input = color_input(hudless.p), backbuffer_input = color_input(backbuffer.p);
        const abi_v2::resource_tag tags[] {{{nullptr, tag_guid, 1}, &hudless_input, 2, 1, {0, 0, width, height}}, {{nullptr, tag_guid, 1}, &backbuffer_input, 53, 1, {0, 0, width, height}}};
        ui_tag_tick = clock.now();
        returned(sl.global_tag(view, tags, 2, ui_choice == ui_form::listed ? command() : nullptr), "the HUD-less and Backbuffer tags");
        if (ui_choice == ui_form::untagged) {
          const abi_v2::resource_tag untags[] {{{nullptr, tag_guid, 1}, nullptr, 2, 1, {}}, {{nullptr, tag_guid, 1}, nullptr, 53, 1, {}}};
          returned(sl.global_tag(view, untags, 2, nullptr), "the HUD-less and Backbuffer untags");
        }
        poll("ui");
      }
      for (std::uint32_t i = 0; i != generated_inside; ++i) {
        generated_present(true);
      }
      if (sr_choice == sr_form::local || sr_choice == sr_form::tag) {
        const auto declared = declared_state(matrix::lifecycle::only_valid_now);
        const auto resource = depth_input(declared);
        const abi_v2::resource_tag input {{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
        const base_structure *inputs[] {&view.base, &input.base};
        if (sr_choice == sr_form::local) {
          decide("sr_local", local_case(declared, evidence_seen, evidence_state, generated_in_evaluate), continuity::sr_evaluate, {"evaluate"});
          immediate(continuity::sr_evaluate);
        }
        // A generated Present inside the evaluation: the add-on's hook ends
        // the input source's present generation once inside the next
        // nominated copy's call, between its input's read and the copy
        // (record_impl), as DLSS-G's present thread does; it is withdrawn
        // when the evaluation nominated no copy (an FG scope that owns the
        // viewport). The Presents themselves follow the evaluation.
        if (generated_in_evaluate) {
          present_before_next_copy();
        }
        const auto copy_before = newest_copy();
        evaluation_began = true;
        returned(sl.evaluate(0, *token, inputs, sr_choice == sr_form::local ? 2 : 1, command()), "slEvaluateFeature");
        note_copy(continuity::sr_evaluate, copy_before);
        if (generated_in_evaluate) {
          cancel_present_before_next_copy();
        }
        poll("evaluate");
        for (std::uint32_t i = 0; i != generated_in_evaluate; ++i) {
          generated_present(true);
        }
      } else if (sr_choice == sr_form::other_queue) {
        // The game's own queue's recording: the depth written there, its
        // input declared as it is there (or undeclared), nothing recorded.
        std::uint64_t native {};
        const auto n = game_work.open_next(native);
        write_depth_pixels(true, game_work.lists[n].p);
        const auto declared = choice == state_form::unprovable ? UINT32_MAX : std::uint32_t(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const auto resource = depth_input(declared);
        const abi_v2::resource_tag input {{nullptr, tag_guid, 1}, &resource, 0, 1, active_area};
        const base_structure *inputs[] {&view.base, &input.base};
        decide("sr_other_queue", local_case(declared, false, 0), continuity::sr_evaluate, {"evaluate"});
        const auto copy_before = newest_copy();
        evaluation_began = true;
        returned(sl.evaluate(0, *token, inputs, 2, reinterpret_cast<void *>(native)), "slEvaluateFeature on the game's queue");
        note_copy(continuity::sr_evaluate, copy_before);
        poll("evaluate");
        // A GPU-bound game: the SR work on the game's queue runs only once
        // the frame's Present returned (open_sr_gate).
        if (of<sr_gpu_form>(value, events::sr_gpu) == sr_gpu_form::after_present) {
          checked(game_work.queue->Wait(sr_gate.p, ++sr_gate_value), "Hold the SR work on the game's queue until after the frame's Present");
          script_order.wait(game_work.queue.p, sr_gate_token, sr_gate_value);
          sr_gate_closed = true;
        }
        // A copy recorded on a game list never executed with a copy before,
        // when no copy of the source was submitted on the game's queue within
        // the source age (after copies on the presenting queue), may have no
        // copy-out storage (the add-on predicts a new list to run where the
        // newest direct-list copy ran, record_impl, and gives copy-out storage
        // to a source read elsewhere within the source age; it takes an idle
        // slot's storage only where one exists: copy_out's unstored): another
        // queue reads it only once its recording retired (the Reset rule,
        // immutable_for_other_queues), never through the order fence nor at
        // its completion alone. Until
        // 10-28 the judge took it as readable once complete, so a fresh
        // process's first SR copy on the game's queue failed as stale_shown
        // (seed c28965dca8d7c76c at frame 11, 3 of 3 runs, copy_out
        // unstored=1), and inside the fixed set, whose earlier timelines had
        // run every list, it passed (the 10-28 review's H5).
        const bool cold = !game_work_carried[n] &&
                          !(other_queue_copy && clock.now() - other_queue_copy < sunshine_freshness::source_age_ms);
        const bool readable = execute_elsewhere(game_work, n, game_order, cold);
        auto completion = game_work.completion();
        if (cold) {
          completion = [complete = std::move(completion), retired = game_work.retired(n)] {
            return complete() && retired();
          };
        }
        recorded_elsewhere(continuity::sr_evaluate, std::move(completion), readable);
        game_work_carried[n] = game_work_carried[n] || (supply.elsewhere & continuity::sr_evaluate);
        other_queue_copy = clock.now();
      }
      // A hitch right before DLSS-G's evaluation of the frame, after its other
      // calls (before_dlssg).
      if (hitch == hitch_form::before_dlssg) {
        clock.hitch(hitch_ms(hitch));
      }
      dlssg_evaluation(of<dlssg_form>(value, events::dlssg));
      // FG as the frame's evaluations saw it, then a late FG Off: the game's
      // options call after those evaluations, before the frame's Present
      // (fg=off_late). Its order fences are renewed after that Present.
      const bool fg_evaluated = fg_on;
      decided_in_scope = frame_generation_scope();
      const bool late_off = fg_on && of<fg_event>(value, events::fg) == fg_event::off_late;
      if (late_off) {
        set_frame_generation(false);
        renew_after_present = true;
      }
      move(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      // A hitch after the frame's calls, before its Present. What follows is
      // the fixture's bookkeeping, read at the time the Present decides:
      // the FG scope's hold of its copy and its compute scope (until the
      // 10-27 review read before the hitch, so SR stayed kept out at a
      // Present where the held FG copy had expired, the review's J2).
      if (hitch == hitch_form::after_sources) {
        clock.hitch(hitch_ms(hitch));
      }
      // A copy deferred to the frame's Present (R2) is made right before
      // that Present, after any hitch here: its made tick is now (until
      // 10-28 its tag call's, so after a hitch the frame's own copy, held at
      // a later generated Present, was judged as old as the hitch, stale).
      // The add-on stamps it there too (present_deferred; until 11-01 at its
      // tag call, so it dropped the held copy that much before the judge's
      // source age: fg-on-deferred-tag-hitch-after-sources-held).
      if (deferred_tag && !cancel_expected) {
        for (unsigned i = 0; i != supply.made.size(); ++i) {
          if ((continuity::tag_copy & (1u << i)) && supply.made[i]) {
            supply.made[i] = clock.now();
          }
        }
      }
      // A frame whose deferred FG tag copy waits for DLSS-G to evaluate its
      // frame, which DLSS-G does not do, supplies that copy as permitted
      // only: DLSS-G evaluated the previous deferred frame, so the add-on
      // cannot tell this frame's Present from an earlier frame's, and the
      // Present holds the earlier copy. Any later frame without DLSS-G's
      // evaluation copies at its Present (known gap C7 until 10-26, when
      // each waited within the source age of any evaluation). DLSS-G's
      // evaluation of a frame without an FG tag of its own copies (known gap
      // C6 until 10-26, when it deferred to an earlier frame's tag copy for
      // the source age).
      const auto dlssg_choice = of<dlssg_form>(value, events::dlssg);
      const auto now = clock.now();
      if (fg_on && deferred_tag && awaits_dlssg && dlssg_choice == dlssg_form::none && dlssg_evaluated && now - dlssg_evaluated < sunshine_freshness::source_age_ms + continuity::tag_slack_ms && (supply.legal & continuity::tag_copy)) {
        supply.legal &= std::uint8_t(~continuity::tag_copy);
        supply.permitted |= continuity::tag_copy;
      }
      // A live FG scope keeps the viewport's SR input out of the frame after
      // one that supplied an FG copy, which it expects to supply too: the
      // Present shows the scope's FG depth instead (FG keeps the frames it
      // supplies; the next frame without an FG copy turns the scope silent).
      const bool owned = std::exchange(fg_supplied, false) && fg_evaluated;
      // So does a live scope whose frame's only FG input is DLSS-G's compute
      // copy: the Present shows that copy's copy-out once the game's fence
      // waits order the presenting queue after it (U-C1), else the scope's
      // newest readable copy (until 10-27 known gap C1: readable only once
      // complete, and before that three DLSS-G evaluations later).
      const bool compute_scope = fg_on && dlssg_choice == dlssg_form::compute && fg_copy && !fg_copy_lost && now - fg_copy < sunshine_freshness::fg_silent_ms + continuity::tag_slack_ms && !(supply.legal & continuity::tag_copy);
      // When the contract does not let the add-on read that copy at this
      // Present (execute_elsewhere: the fence it waits on is refused, or the
      // CPU did not wait), the scope shows its held copy, the SR input still
      // required as an owned source (until 10-27 it was permitted there).
      // A silent scope (the frame before supplied no FG copy) whose display
      // holds its own FG copy, made since FG On and younger than the source
      // age, keeps the viewport's SR copies off-screen until that copy
      // expires (frame_generation_holds, which awaits the owner's decision):
      // like an owned frame's, they leave no copy a later Present could hold.
      // Not once a late signal reached the fence that copy was read through
      // (fg_copy_lost): the add-on drops it and the SR copies stand in
      // (until the 10-27 review the fixture kept them out for the source
      // age, which excused a mono Present and hid stale SR depth). That
      // copy's age counts from fg_copy: this bookkeeping's clock read, or
      // for a tag copy deferred to its frame's Present (R2) the add-on's own
      // stamp of it there (run_frame, after that Present). Until 11-02 it
      // was this read, before that Present, so where the clock ticked
      // between the two the fixture ended the hold a clock period before
      // the add-on did and judged the add-on's hold at its last Present
      // stale_shown (fixed-set seed dda773a575be6561 at frame 172, 1 run in
      // 6; fg-on-deferred-tag-hitch-before-present-held).
      const bool held_off = fg_on && !owned && fg_copy && !fg_copy_lost && fg_copy >= fg_enabled_at &&
                            now - fg_copy < sunshine_freshness::source_age_ms;
      // An FG copy its scope accepts: an FG tag's, or DLSS-G's in a state
      // its list recorded (its compute copy is accepted later, with its own
      // tick).
      if (fg_on && frame_generation_scope() && ((supply.legal | supply.permitted | supply.limited) & (continuity::tag_copy | continuity::dlssg_input))) {
        fg_copy = now;
        fg_supplied = true;
        fg_copy_lost = false;
        fg_copy_relied.clear();
        fg_copy_capture = std::max(supply.capture[1], supply.capture[2]) & ~continuity::copy_fg_input;
        for (const auto &value : supply.completions) {
          if (value.sources == continuity::dlssg_input && value.relied) {
            fg_copy_relied.push_back(value.relied);
          }
        }
      }
      if (fg_evaluated && dlssg_choice != dlssg_form::none) {
        dlssg_evaluated = now;
        ++dlssg_frames;
      }
      // A late FG Off retired the frame's FG role before any Present showed
      // its FG copies. The add-on keeps them, published after the copy the
      // lineage's display showed, as the frame's own depth of the display
      // lineage, the viewport's ordinary source (retire_source at a confirmed
      // Off, owner ruling 10-30 Q3; depth_capture::frame_generation_retired):
      // its FG tag's copy, made at the call or deferred to its Present (R2)
      // and made there after the Off, and DLSS-G's copy, made at its
      // evaluation. They supply that Present as permitted, never required,
      // each aged from where it was made (supply::made, where the add-on
      // stamps it), and the lineage's SR successor is required at its first
      // readable frame as any. A tag copy the frame's untag cancels (R7) is
      // never made. Until 11-01 only the deferred tag copy supplied it (until
      // 10-28 the add-on made that copy the FG view again beside the SR
      // view, which held it stale; the fixture called it a source with no
      // contract), while the add-on has kept DLSS-G's copy since 11-01's
      // deferred fixes: the judge aged the add-on's hold of it from the frame
      // before (fixed-set seed 5d48a3d414f1de10 failed stale at frame 39, age
      // 296 ms, about 1 run in 3), and showed after a hitch since its
      // evaluation it failed as unsupplied
      // (fg-off-late-kept-dlssg-copy-hitch-after-sources).
      if (late_off) {
        constexpr auto fg_copies = std::uint8_t(continuity::tag_copy | continuity::dlssg_input);
        const auto kept = std::uint8_t((supply.legal | supply.limited | supply.permitted) & fg_copies &
          ~(cancel_expected ? continuity::tag_copy : 0));
        for (auto *mask : {&supply.legal, &supply.limited, &supply.permitted, &supply.elsewhere, &supply.immediate}) {
          *mask = std::uint8_t(*mask & ~fg_copies);
        }
        for (const unsigned i : {1u, 2u}) {
          if (!(kept & (1u << i))) {
            supply.made[i] = 0;
          }
          supply.capture[i] = 0;
        }
        supply.completions.erase(std::remove_if(supply.completions.begin(), supply.completions.end(), [](const auto &value) {
          return (value.sources & fg_copies) != 0;
        }), supply.completions.end());
        supply.permitted |= kept;
      }
      const auto owned_sr = owned || held_off || compute_scope ? std::uint8_t(supply.legal & continuity::sr_evaluate) : std::uint8_t {};
      // A frame whose Present can read none of its legal sources: its copies
      // on another queue are not readable there (execute_elsewhere), and its
      // SR input, if any, a live FG scope keeps out (owned). After a late
      // signal the copies read through that fence are unholdable
      // (signal_again), so O1's drop of a held copy is judged by what else
      // the Present could hold, never excused by a window (until 10-27
      // R1-late-signaller tagged every legal source of every real frame in
      // one, and a frame's own Present before its late signal too). The
      // generated Presents of a frame whose fence a late signal reaches
      // after its Present may drop the hold (late_after, R1).
      const bool unordered = !readable_here && (supply.legal & supply.elsewhere) &&
                             !(supply.legal & std::uint8_t(~supply.elsewhere) & std::uint8_t(~owned_sr));
      const auto now_late = clock.now();
      // The frame's supply: a frame whose legal sources all meet a known
      // limitation is tagged with it; any other legal source must show.
      continuity_log.pending = {};
      continuity_log.pending.legal = std::uint8_t(supply.legal | supply.limited);
      continuity_log.pending.permitted = supply.permitted;
      continuity_log.pending.limited = supply.limited;
      continuity_log.pending.owned = owned_sr;
      // Copied, yet kept off-screen by the FG hold (held_off; a live scope
      // refuses the others): after FG Off the lineage's successor copies.
      const bool fg_held = held_off && !owned && !compute_scope;
      continuity_log.pending.fg_held = fg_held ? owned_sr : std::uint8_t {};
      // The frame's Present comes with FG off (also after a late Off).
      continuity_log.pending.fg_off = !fg_on;
      continuity_log.pending.limit = supply.legal ? continuity::limitation::none : supply.limit;
      // Frame identity (ui_unidentified): a frame with a depth call, or one
      // with FG off, begins a window of its own at its real Present; a
      // callless one counts as a Present of the earlier frame's. Its supply
      // (none) meets OPEN C's depth twin while that window still counts it.
      const auto fg_change = of<fg_event>(value, events::fg);
      const bool depth_called = form != tag_form::none || sr_choice != sr_form::none ||
        dlssg_choice != dlssg_form::none || fg_change != fg_event::keep;
      if (!fg_on || depth_called) {
        ui_window_frame = frame;
        ui_window_presents = 0;
        ui_unidentified = false;
      } else {
        ui_unidentified = ++ui_window_presents <= declared_generated;
      }
      if (ui_unidentified && !supply.legal) {
        continuity_log.pending.limit = continuity::limitation::unidentified_real;
      }
      continuity_log.pending.tick = supply.tick;
      continuity_log.pending.late_after = std::any_of(late_signals.begin(), late_signals.end(), [&](const order_fence *order) {
        return late_relied(*order, now_late);
      });
      continuity_log.pending.unordered = unordered;
      continuity_log.pending.immediate = supply.immediate;
      continuity_log.pending.made = supply.made;
      continuity_log.pending.capture = supply.capture;
      // DLSS-G adds this frame's generated Presents (judge::present, between).
      continuity_log.pending.fg_frame = fg_supplied;
      // An untag that no tag call of the frame follows (R7: after the frame's
      // tag, by its token or a global untag, or a global untag in a frame
      // without a tag) may end the source's hold: the judge does not require
      // this frame's Present to hold depth (strict_holds). An untag before
      // the frame's own tag call ends nothing, since that call names the
      // depth again, so that Present is judged as any. Until 11-01 every
      // untag form excused it, which passed a mono Present while the frame's
      // deferred tag copy was holdable
      // (fg-on-deferred-tag-hitch-after-sources-held-untag-before).
      const bool tagged_again = untag == untag_form::before_global && form != tag_form::none;
      continuity_log.pending.drop_permitted = untag != untag_form::none && !tagged_again;
      // The freshness rule: the frame's copies recorded on another queue,
      // readable once complete, except SR copies a live FG scope keeps out as
      // it keeps the presenting queue's own (owned), and copies of a source
      // the presenting queue does not show: FG input without an FG scope.
      continuity_log.pending.elsewhere = supply.elsewhere;
      for (const auto &value : supply.completions) {
        if (value.sources == continuity::sr_evaluate && (owned || compute_scope)) {
          continue;
        }
        if (value.sources == continuity::dlssg_input && !(fg_on && frame_generation_scope())) {
          continue;
        }
        continuity_log.pending.completes(value.sources, value.probe, value.relied, value.capture,
          value.sources == continuity::sr_evaluate && fg_held);
      }
      deferred_copy_before = deferred_tag && !cancel_expected && (supply.legal & continuity::tag_copy) &&
        !supply.capture[1] ? std::max<std::uint64_t>(newest_copy(), 1) : 0;
      // A pause after the bookkeeping, before the frame's real Present.
      if (hitch == hitch_form::before_present) {
        clock.hitch(hitch_ms(hitch));
      }
    }

    void dlssg_evaluation(frame_events::dlssg_form form) {
      using frame_events::dlssg_form;
      if (form == dlssg_form::none) {
        return;
      }
      // The game's native list, as DLSS-G records on it.
      const auto native_list = [this] {
        return reinterpret_cast<ID3D12GraphicsCommandList *>(game_native_command);
      };
      if (!feature) {
        require(ngx_create(native_list(), 11, &dlssg_inputs, &feature) == ngx_fixture::success && feature, "DLSS-G feature creation failed");
      }
      dlssg_inputs.set_ui("DLSSG.DepthSubrectBaseX", active_area.left);
      dlssg_inputs.set_ui("DLSSG.DepthSubrectBaseY", active_area.top);
      dlssg_inputs.set_ui("DLSSG.DepthSubrectWidth", active_area.width);
      dlssg_inputs.set_ui("DLSSG.DepthSubrectHeight", active_area.height);
      matrix::ngx_case value;
      value.feature = matrix::ngx_feature::frame_generation;
      value.subrect = matrix::ngx_subrect::inside;
      constexpr auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      const bool observed = form == dlssg_form::direct || form == dlssg_form::compute;
      if (form == dlssg_form::compute || form == dlssg_form::compute_unobserved) {
        // DLSS-G's own compute recording after the game's work; the
        // presenting queue waits for its signal and reads its copy-out at the
        // next Present (U-C1), or its copy once DLSS-G reset that ring entry.
        move(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        submit(false);
        begin_commands();
        evidence_seen = false;
        std::uint64_t native {};
        const auto n = dlssg_work.open_next(native);
        auto *compute = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
        if (observed) {
          transition(compute, selected->resource.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read);
        }
        value.list = matrix::list_type::compute;
        value.depth = recorded(observed, read);
        decide("dlssg_compute", value, continuity::dlssg_input, {"dlssg"}, true);
        const auto copy_before = newest_copy();
        evaluation_began = true;
        require(ngx_evaluate(compute, feature, &dlssg_inputs, nullptr) == ngx_fixture::success, "DLSS-G's compute evaluation changed the SDK result");
        note_copy(continuity::dlssg_input, copy_before);
        poll("dlssg");
        if (observed) {
          transition(compute, selected->resource.p, read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        // A GPU-bound game: DLSS-G's compute work runs only once the
        // frame's Present returned (open_dlssg_gate).
        if (frame_events::of<frame_events::dlssg_gpu_form>((*frames)[frame], events::dlssg_gpu) == frame_events::dlssg_gpu_form::after_present) {
          checked(dlssg_work.queue->Wait(dlssg_gate.p, ++dlssg_gate_value), "Hold DLSS-G's compute work until after the frame's Present");
          script_order.wait(dlssg_work.queue.p, dlssg_gate_token, dlssg_gate_value);
          dlssg_gate_closed = true;
        }
        // Streamline's present queue waits for DLSS-G's compute signal, so
        // the Presents after this one are ordered after its copy-out (U-C1),
        // as the frame's fence events have it.
        const bool readable = execute_elsewhere(dlssg_work, n, dlssg_order);
        recorded_elsewhere(continuity::dlssg_input, dlssg_work.completion(), readable);
        if (observed) {
          other_queue_copy = clock.now();
        }
        return;
      }
      auto *list = native_list();
      if (observed) {
        move(read);
        transition(list, hudless.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, read);
        transition(list, backbuffer.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, read);
      }
      value.depth = recorded(evidence_seen, evidence_state);
      decide("dlssg_direct", value, continuity::dlssg_input, {"dlssg"}, true);
      immediate(continuity::dlssg_input);
      const auto copy_before = newest_copy();
      evaluation_began = true;
      require(ngx_evaluate(list, feature, &dlssg_inputs, nullptr) == ngx_fixture::success, "DLSS-G's evaluation changed the SDK result");
      note_copy(continuity::dlssg_input, copy_before);
      poll("dlssg");
      if (observed) {
        transition(list, hudless.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        transition(list, backbuffer.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }
    }

    // The FG options with the generated count the game declares
    // (numFramesToGenerate, and DLSS-G's MultiFrameCount), as the add-on
    // reads it (frame_generation_policy::generated_frames). A game repeats
    // its options; a repeat keeps FG's enable time.
    void set_frame_generation(bool on, std::uint32_t generated = 0) {
      if (generated) {
        declared_generated = generated;
      }
      const fg_options value {{nullptr, fg_options_type, 3}, on ? 1u : 0u, on ? declared_generated : 0u};
      returned(sl.set_options(view, value), "slDLSSGSetOptions");
      dlssg_inputs.set_ui("DLSSG.MultiFrameCount", declared_generated);
      if (on && !fg_on) {
        fg_enabled_at = clock.now();
      }
      fg_on = on;
    }

    // The generated count the FG options declared last.
    std::uint32_t declared_generated {1};

    // The frame's FG options event made inside its real Present
    // (fg_at=depth_read): armed once the frame's calls are made, made by the
    // add-on's call at that Present's depth frame.
    struct options_at_present {
      bool armed {}, made {}, on {};
      std::uint32_t generated {};
    } pending_options;

    static void make_pending_options(void *context) {
      auto &self = *static_cast<timeline_fixture *>(context);
      self.pending_options.made = true;
      self.set_frame_generation(self.pending_options.on, self.pending_options.generated);
    }

    // The add-on's metadata reads fail while on is set during a frame with
    // reads=lost_at_present (its Presents after its calls).
    void lose_reads(bool on) {
      using namespace frame_events;
      if (of<reads_form>((*frames)[frame], events::reads) == reads_form::lost_at_present) {
        fail_metadata_reads(on ? ~0u : 0u);
      }
    }

    // Once the frame's calls are made, right before its real Present.
    // Engaged, the work the frame gave its rings completes first (their
    // done fences and each ring's own order value, wait_rings), except work
    // a gate holds until after the Present (sr_gpu or dlssg_gpu
    // after_present), the in-flight cases: with the default (prompt) the
    // add-on's read at that Present raced the GPU's completion (the
    // regression rows' note above), which the frame boundary's barrier
    // after the Present comes too late to settle (owner decision C3b,
    // 11-02). --wall-clock keeps that race.
    void before_real_present() {
      if (clock.engaged()) {
        clock.wait_rings("before the frame's Present");
      }
      if (pending_options.armed) {
        options_at_depth_frame(&timeline_fixture::make_pending_options, this);
      }
      lose_reads(true);
    }

    // The FG mode each Present used for UI and for depth (the add-on's
    // SunshineGame3DTestPresentFrameGeneration). In a timeline with an FG
    // options event inside a Present (fg_at=depth_read) both must be the
    // same on every Present where depth does not revoke FG input: the UI's
    // FG mode (known and enabled) and depth's (its policy requires FG); and
    // every consumer must find the FG snapshot its Present took (no
    // fallback since the run began).
    bool judge_modes {};
    std::uint64_t fallbacks_before {};

    void judge_present_modes(const char *which) {
      if (!judge_modes) {
        return;
      }
      unsigned ui {}, depth {};
      std::uint64_t fallbacks {};
      require(present_frame_generation(observed.runtime, &ui, &depth, &fallbacks), "The add-on did not report a Present's FG modes");
      if (trace) {
        std::printf("TRACE frame=%u fg_modes %s ui=%u depth=%u fallbacks=%llu\n", frame, which, ui, depth, static_cast<unsigned long long>(fallbacks - fallbacks_before));
      }
      if (fallbacks != fallbacks_before) {
        fail("fg_snapshot:fallback", frame_text(frame) + ": " + which + " had a consumer that found no FG snapshot of its Present", frame, "fallback");
        fallbacks_before = fallbacks;
      }
      const bool ui_fg = (ui & 3u) == 3u, depth_fg = (depth & 2u) != 0;
      if (!(depth & 4u) && ui_fg != depth_fg) {
        fail("fg_modes:disagree", frame_text(frame) + ": " + which + " used FG " + (ui_fg ? "on" : "off") + " for UI and " + (depth_fg ? "on" : "off") + " for depth", frame, std::string("ui_") + (ui_fg ? "on" : "off"));
      }
    }

    // PD7 (11-01): the UI holds the decision of the last Present that
    // offered it a Streamline UI tag on a later Present without input of
    // its own (a UI row held, generated_present) only where that later
    // Present presents the same frame as the oracle gives Presents to
    // frames: a real Present its frame; a generated Present after it, or
    // inside the next frame's recording before the add-on began an
    // evaluation of that frame, the last real Present's frame; one after
    // that evaluation began, the frame being recorded (as depth's window
    // ends at the next evaluation, frame_presenting). Until 11-01 the UI
    // counted its own window of the declared generated count, so the next
    // frame's Presents inside it held the earlier frame's decision. Judged
    // where depth presented a frame at that Present (its acquisition's
    // presenting floor): without that window the UI still counts
    // (ui_evidence::generated_in_window), a limit of no frame identity, not
    // of this decision. Both ways (11-01 final review F13): a hold on another
    // frame is ui_frame:held, and a generated Present that presents that
    // same frame and offers no input of its own must hold it (a UI row held,
    // generated_present or same_snapshot_inputs), else ui_frame:dropped.
    // Its record tells a missing hold from a withdrawn one: the add-on's
    // hold turns each UI row it found absent (filtered, nothing offered)
    // into held, and a qualification that withdrew what a Present offered or
    // held (its scope changed, or its last observation is older than the
    // source age) refuses those rows as qualification_unavailable, which
    // ui_qualification owns. So a Present with an absent UI row and no row
    // ready, held, or withdrawn that way made no hold. Depth's window, and so
    // the hold, reaches a generated Present only for a frame that supplied
    // an FG copy its live scope accepts (frame_presenting counts from its FG
    // stamps), and ends after the declared generated count of Presents that
    // present the frame after its real Present (the Present's grant): a
    // further one (a frame whose generated Presents came after its real
    // Present and again inside the next frame's recording) is not judged
    // so (FG On's first frame: the frame before it has no FG copy, and depth
    // presents no frame at its generated Presents). Nor is one at or after
    // the UI's evidence age since the last UI tag call (the UI scope,
    // query_ui_scope, ends there and the UI shows no mask until the next
    // tag: a limit of the evidence age, not of this decision; a game that
    // stopped tagging UI, or a pause between a real Present and its
    // generated Presents, where depth still presents the frame). Until
    // 11-01 only the first way was judged, so a depth window that ended
    // early, or a UI that compared the wrong window, dropped every such hold
    // unseen.
    unsigned ui_input_frame {UINT_MAX};
    bool ui_input_window {};
    std::uint64_t ui_input_floor {};  // Its floor (--trace only).
    // Right before the run's last UI tag call (its HUD-less and Backbuffer
    // tags), 0 before the first.
    std::uint64_t ui_tag_tick {};
    // The last real Present's frame supplied an FG copy its live scope
    // accepts (fg_supplied at that Present).
    bool ui_presented_fg {};
    std::uint64_t ui_judged_present {UINT64_MAX};
    // The generated Presents that presented the last real Present's frame
    // since that Present (between).
    std::uint32_t generated_since_real {};
    // Frame identity for the add-on (11-01, OPEN C). With FG on, a frame
    // that makes no depth call before its real Present (no tag call, SR or
    // DLSS-G evaluation, no FG On or Off event; its Streamline constants,
    // UI tag calls and its own count in the FG options are none) gives the
    // add-on no frame of its own: at its real Present, and at every Present
    // after it until a later frame begins an evaluation, the add-on sees
    // exactly what it sees at the generated Presents of the last real
    // Present's frame that had one (ui_window_frame) where DLSS-G presents
    // all of them (a frame's constants and UI tag calls end no window:
    // DLSS-G presents the earlier frame's generated Presents inside the next
    // frame's recording before its evaluation, owner ruling Q2 and F13). So
    // while the Presents since that real Present (ui_window_presents: the
    // callless frames' real Presents and every generated Present between)
    // are within the generated count in force at the Present (the add-on's
    // grant reads the count each Present's FG snapshot holds, a callless
    // frame's own count included, so the judge does too), depth's window
    // still presents that frame and the UI may hold its decision, as the T1
    // grace would reuse it. Soak seed 117a6b99131447b1 at frame 40 (its
    // frame 39 declared one generated Present and DLSS-G presented none)
    // makes the same calls as the 4-frame reproducer of
    // fg-on-deferred-tag-generated-inside-after-hitch-untag at its frame 3,
    // whose generated Presents must hold. Such a hold of that frame's
    // decision is not judged ui_frame:held, a limit of no frame identity
    // like a Present without a depth window at its input (counted:
    // ui_unidentified_holds). Until the second fix review of 11-01 only the
    // callless frame's real Present was left out, so its own generated
    // Presents within the earlier frame's count still failed. Its depth
    // twin, the earlier frame's copy held past the source age at such a
    // real Present, is the known gap OPEN-C-depth-unidentified-real
    // (limitation::unidentified_real), an accepted limit (owner ruling
    // 11-02).
    unsigned ui_window_frame {UINT_MAX};
    std::uint32_t ui_window_presents {};
    // A callless real Present came since ui_window_frame's within its count
    // (frame_body), until the next real Present of a frame with identity.
    bool ui_unidentified {};
    // This Present is such a Present (the real one, or a generated one
    // between within the count).
    bool ui_unidentified_now {};

    // generated: 0 for a real Present, else the generated Present's place
    // among those of its frame (generated_since_real; 1 where it presents
    // the frame being recorded).
    void judge_ui_frame(const char *which, unsigned shown_frame, std::uint32_t generated) {
      if (record_history.records.empty()) {
        return;
      }
      const auto &r = record_history.records.back();
      if (r.present == ui_judged_present) {
        return;
      }
      ui_judged_present = r.present;
      using sunshine_game3d::source_slot;
      using sunshine_game3d::source_state;
      bool offered {}, held {}, input {}, kept {}, absent {};
      const auto reason_is = [&](source_slot slot, const char *reason) {
        return r[slot].reason && !std::strcmp(r[slot].reason, reason);
      };
      for (const auto slot : {source_slot::ui_alpha, source_slot::ui_color, source_slot::backbuffer}) {
        offered = offered || r[slot].state == source_state::ready;
      }
      for (const auto slot : {source_slot::ui_alpha, source_slot::ui_color, source_slot::backbuffer, source_slot::hudless, source_slot::ui_layer}) {
        held = held || (r[slot].state == source_state::held && reason_is(slot, "generated_present"));
        kept = kept || (r[slot].state == source_state::held && (reason_is(slot, "generated_present") || reason_is(slot, "same_snapshot_inputs")));
        input = input || r[slot].state == source_state::ready || (r[slot].state == source_state::refused && reason_is(slot, "qualification_unavailable"));
        absent = absent || r[slot].state == source_state::absent;
      }
      const bool same_frame = generated && generated <= declared_generated && ui_input_window && shown_frame == ui_input_frame;
      // Judged where depth's window reaches its generated Presents (the
      // presented frame supplied an FG copy its live scope accepts) and the
      // UI's evidence is fresh (a UI tag call within the evidence age: the
      // UI scope, query_ui_scope, ends after it).
      const bool fresh_evidence = ui_tag_tick && clock.now() - ui_tag_tick < sunshine_freshness::evidence_age_ms;
      const bool due = same_frame && ui_presented_fg && fresh_evidence;
      result.ui_holds += due && held;
      if (trace && (offered || held || same_frame)) {
        // Each UI row (UIAlpha, UI color, Backbuffer, HUD-less, layer) and
        // depth's presenting floor at this Present and at the tag's.
        std::string rows;
        for (const auto slot : {source_slot::ui_alpha, source_slot::ui_color, source_slot::backbuffer, source_slot::hudless, source_slot::ui_layer}) {
          rows += std::string(rows.empty() ? "" : ",") + sunshine_game3d::name(r[slot].state) + "/" + (r[slot].reason ? r[slot].reason : "-");
        }
        sunshine_streamline::depth_capture::depth_choice now_choice {};
        const auto floor = queue_choice(observed.runtime, &now_choice) ? now_choice.decision.presenting_floor : 0;
        std::printf("TRACE frame=%u ui_frame %s present=%llu shown=%u input=%d window=%u offered=%u held=%u kept=%u absent=%u own_input=%u nth=%u floor=%llu input_floor=%llu tag_age=%lld fg=%u fg_frame=%u due=%u unidentified=%u window_frame=%d window_presents=%u rows=%s\n", frame, which, static_cast<unsigned long long>(r.present), shown_frame, ui_input_frame == UINT_MAX ? -1 : int(ui_input_frame), unsigned(ui_input_window), unsigned(offered), unsigned(held), unsigned(kept), unsigned(absent), unsigned(input), generated, static_cast<unsigned long long>(floor), static_cast<unsigned long long>(ui_input_floor), ui_tag_tick ? static_cast<long long>(clock.now() - ui_tag_tick) : -1ll, unsigned(fg_on), unsigned(ui_presented_fg), unsigned(due), unsigned(ui_unidentified_now), ui_window_frame == UINT_MAX ? -1 : int(ui_window_frame), ui_window_presents, rows.c_str());
      }
      if (offered) {
        sunshine_streamline::depth_capture::depth_choice choice {};
        ui_input_frame = shown_frame;
        ui_input_window = queue_choice(observed.runtime, &choice) && choice.decision.presenting_floor;
        ui_input_floor = ui_input_window ? choice.decision.presenting_floor : 0;
      } else if (held && ui_input_window && shown_frame != ui_input_frame && ui_unidentified_now &&
          ui_input_frame == ui_window_frame) {
        ++result.ui_unidentified_holds;
      } else if (held && ui_input_window && shown_frame != ui_input_frame) {
        fail("ui_frame:held", frame_text(frame) + ": " + which + " held frame " + std::to_string(ui_input_frame) + "'s UI decision as a generated Present of it, and it presents frame " + std::to_string(shown_frame), frame, "generated_present");
      } else if (due && absent && !input && !kept) {
        fail("ui_frame:dropped", frame_text(frame) + ": " + which + " presents frame " + std::to_string(shown_frame) + ", whose Present offered the last UI tag, offered no UI input of its own and did not hold that Present's UI decision", frame, "generated_present");
      }
    }

    void run_frame() {
      using namespace frame_events;
      const auto &value = (*frames)[frame];
      const auto event = of<fg_event>(value, events::fg);
      const auto generated = std::uint32_t(value[events::generated]);
      const bool options_at_present = (event == fg_event::on || event == fg_event::off) &&
        of<fg_at_form>(value, events::fg_at) == fg_at_form::depth_read;
      pending_options = {options_at_present, false, event == fg_event::on, generated};
      if (options_at_present) {
        // Made inside the frame's real Present (before_real_present); its
        // fences are renewed after it.
      } else if (event == fg_event::on || event == fg_event::off) {
        set_frame_generation(event == fg_event::on, generated);
        // The game creates its frame-generation objects anew, these fences
        // among them.
        renew_order();
      } else if (fg_on && generated && generated != declared_generated) {
        // The frame's generated count, declared before its calls.
        set_frame_generation(true, generated);
      }
      if (value[events::drs]) {
        area = (area + 1) % 4;
        const unsigned w = selected->width, h = selected->height;
        const sunshine_streamline::extent areas[] {{0, 0, w, h}, {0, 0, w, h - 3}, {0, 0, w * 7 / 8, h * 7 / 8}, {0, 0, w * 3 / 4, h * 3 / 4}};
        active_area = areas[area];
      }
      supply = {};
      decisions.clear();
      polled.clear();
      refusal_serial = last_refusal().serial;
      render_tracked_depth = [this] {
        frame_body();
        before_real_present();
      };
      after_present = [this] {
        open_dlssg_gate();
        open_sr_gate();
      };
      deferred_copy_before = 0;
      present_frame();
      const auto presented = clock.now();
      lose_reads(false);
      render_tracked_depth = {};
      after_present = {};
      if (pending_options.armed) {
        require(pending_options.made, "The frame's FG options event was not made inside its real Present");
        pending_options = {};
        renew_order();
      }
      judge_present_modes("its Present");
      generated_since_real = 0;
      ui_presented_fg = fg_supplied;
      // No frame identity here (ui_unidentified, frame_body).
      ui_unidentified_now = ui_unidentified;
      judge_ui_frame("its Present", frame, 0);
      ui_unidentified_now = false;
      poll("present");
      // The frame's tag copy deferred to this Present, published by it.
      std::uint64_t deferred_copy {};
      if (deferred_copy_before) {
        if (const auto after = newest_copy(); after && after != deferred_copy_before) {
          supply.capture[1] = after;
          deferred_copy = after & ~continuity::copy_fg_input;
          continuity_log.judge.learned_capture(continuity::tag_copy, after);
        }
      }
      // That copy is the FG copy the frame supplied its scope (fg_supplied),
      // made inside this Present, where the add-on stamps it
      // (present_deferred, docs/reshade-sbs.md#deferred-copy-tick): the FG
      // hold's window (fg_copy, held_off) starts at that stamp, the record's
      // depth tick where this Present shows the copy, which must lie between
      // the bookkeeping's clock read before the Present and the read after
      // it (else the add-on stamped the copy elsewhere); where the Present
      // shows another copy, at the read after it (never early, late by at
      // most the rest of that Present's step). Until 11-02 the window began
      // at the bookkeeping's read (held_off).
      if (deferred_copy && fg_supplied) {
        auto tick = presented;
        if (!record_history.records.empty()) {
          const auto &shown = record_history.records.back();
          if (shown.depth_owner == sunshine_game3d::source_slot::fg_depth && shown.depth_capture == deferred_copy) {
            if (shown.depth_tick_ms < fg_copy || shown.depth_tick_ms > presented) {
              fail("oracle:deferred_copy_tick", frame_text(frame) + ": its Present showed its tag copy deferred there (capture " + std::to_string(deferred_copy) + ") stamped at tick " + std::to_string(shown.depth_tick_ms) + ", outside that Present (the clock read " + std::to_string(fg_copy) + " before it and " + std::to_string(presented) + " after it)", frame, "fg_source");
            } else {
              tick = shown.depth_tick_ms;
            }
          }
        }
        fg_copy = tick;
        fg_copy_capture = deferred_copy;
      }
      if (!record_history.records.empty()) {
        present_frames[record_history.records.back().present] = frame;
        // The scope's FG copy is lost to a late signal only where the add-on
        // read it through that fence (a Present read it ordered, possibly in
        // flight): read complete, at its own Present or a later one (a
        // compute copy in flight at its own), it relies on no fence and stays
        // held. Until 10-28 the fixture predicted the read from the fence
        // alone, so where DLSS-G's copy completed before the Present that
        // read it a conflict passed the add-on's kept hold as a dropped one,
        // and the lineage judge then flagged the hold (the 10-28 review's
        // J1; the extra shard's seed d0dd7845ac141ad2 at frame 113).
        const auto &shown = record_history.records.back();
        if (fg_copy_capture && shown.depth_owner == sunshine_game3d::source_slot::fg_depth &&
            shown.depth_capture == fg_copy_capture &&
            shown.depth_read_order != sunshine_game3d::read_order::none &&
            shown.depth_read_order != sunshine_game3d::read_order::ordered) {
          fg_copy_relied.clear();
        }
      }
      // The frame's late signals, after its Present (and before the
      // generated Presents after it).
      for (auto *order : std::exchange(late_signals, {})) {
        signal_again(*order, order->value % 2 == 0, false);
      }
      // A late FG Off: the game creates its frame-generation objects anew,
      // these fences among them.
      if (std::exchange(renew_after_present, false)) {
        renew_order();
      }
      if (supply.legal || supply.limited || (supply.permitted & continuity::api_sources)) {
        api_supplied = clock.now();
      }
      judge_decisions();
      if (trace) {
        for (const auto &value : polled) {
          std::printf("TRACE frame=%u refusal %s at %s adapter=%u viewport=%u class=%u gate=%u stage=%u fg=%u\n", frame, value.value.name, value.step, unsigned(value.value.provider), value.value.viewport, unsigned(value.value.reason), unsigned(value.value.where), unsigned(value.value.stage), unsigned(value.value.frame_generation));
        }
      }
      judge_shown("its Present");
      presented_supply = supply;
      // A hitch after the frame's real Present, before its generated
      // Presents after it.
      if (of<hitch_form>(value, events::hitch) == hitch_form::after_present) {
        clock.hitch(hitch_ms(hitch_form::after_present));
      }
      for (unsigned i = 0, n = unsigned(value[events::generated]); i != n && of<generated_at>(value, events::at) == generated_at::between; ++i) {
        generated_present(false);
      }
    }

    // Retires the run's viewport (FG off, the depth untagged, retired_viewports
    // scratch viewports' constants), presents past the association with no SDK call
    // or draw, then retires the rings' recordings and resets calibration.
    void isolate() {
      using namespace sunshine_streamline;
      // Isolation retires the run's viewport: its Presents may drop the hold.
      continuity_log.drops_permitted = true;
      render_tracked_depth = {};
      if (fg_on) {
        set_frame_generation(false);
      }
      const abi_v2::resource_tag untag {{nullptr, tag_guid, 1}, nullptr, 0, 1, {}};
      returned(sl.global_tag(view, &untag, 1, nullptr), "the isolating untag");
      render_tracked_depth = [this] {
        abi_v2::frame_token *scratch_token {};
        const auto index = ++fg_frame;
        require(sl.new_token(scratch_token, &index) == 0 && scratch_token, "The interposer did not supply a frame token");
        for (unsigned i = 0; i != retired_viewports; ++i) {
          const abi_v2::viewport scratch {{nullptr, viewport_guid, 1}, next_scratch++};
          returned(sl.constants(camera_constants(false), *scratch_token, scratch), "slSetConstants");
        }
      };
      present_frame();
      render_tracked_depth = {};
      for (const auto until = clock.now() + sunshine_freshness::association_ms + 200; clock.now() < until;) {
        present_frame();
      }
      drain();
      require(recalibrate(observed.runtime), "Isolation: SunshineGame3DTestRecalibrate failed");
      active_area = {0, 0, selected->width, selected->height};
      area = 0;
    }

    // -- One timeline --------------------------------------------------------

    // The state of a run of frames (a timeline, or the lifecycle fixture's
    // stretch) on a viewport: its result, its supply windows and a fresh
    // judge. The viewport's FG option is the caller's.
    void begin_run(const timeline_case &value, const std::string &label, std::uint32_t viewport) {
      result = {};
      frames = &value.frames;
      view = {{nullptr, sunshine_streamline::viewport_guid, 1}, viewport};
      last_capture = 0;
      api_supplied = other_queue_copy = fg_copy = dlssg_evaluated = dlssg_frames = deferred_dlssg_frames = 0;
      deferred_known = false;
      fg_supplied = false;
      fg_copy_lost = false;
      fg_copy_relied.clear();
      fg_copy_capture = 0;
      fg_enabled_at = 0;
      present_frames.clear();
      judge_modes = std::any_of(value.frames.begin(), value.frames.end(), [](const generator::row &row) {
        return frame_events::of<frame_events::fg_at_form>(row, frame_events::fg_at) == frame_events::fg_at_form::depth_read;
      });
      pending_options = {};
      if (judge_modes) {
        unsigned ui {}, depth {};
        require(present_frame_generation(observed.runtime, &ui, &depth, &fallbacks_before), "The add-on did not report its FG snapshot fallbacks");
      }
      where = label;
      continuity_log.judge = {};
      // The timelines mark each frame whose events may end its source's
      // hold, so a frame with no legal source must hold within the source age.
      continuity_log.judge.strict_holds = true;
      continuity_log.pending = {};
      continuity_log.drops_permitted = false;
      continuity_log.where = where.c_str();
      declared_generated = 1;
      presented_supply = {};
      // A new viewport: the UI holds no earlier run's decision (its input
      // Present is the scope's, ui_evidence), so neither does the UI judge.
      ui_input_frame = UINT_MAX;
      ui_input_window = false;
      ui_input_floor = 0;
      ui_tag_tick = 0;
      ui_presented_fg = false;
      generated_since_real = 0;
      ui_window_frame = UINT_MAX;
      ui_window_presents = 0;
      ui_unidentified = ui_unidentified_now = false;
      // A new viewport: no read of this run names an earlier run's copy, and
      // its order fences are new, so a run's fence events are its own.
      script_order.clear();
      // The run's presenting queue: the add-on lifecycle fixture presents a
      // later game device's runs on that device's queue.
      script_presenting = queue.p;
      renew_order();
      late_signals.clear();
      renew_after_present = false;
      script_name = where.c_str();
      script_deadline = clock.wall_now() + 15000 + 150 * value.frames.size();
    }

    run_result run(const timeline_case &value, const std::string &label) {
      const auto started = clock.wall_now();
      begin_run(value, label, next_viewport++);
      // The timeline's per-Present trace (record_present): its frames,
      // written after its last one, before isolation presents for a time.
      game3d_present_trace::out.open(runtime_directory / ("trace_" + game3d_present_trace::file_part(label) + ".txt"), "native");
      fg_on = false;
      for (frame = 0; frame != value.frames.size(); ++frame) {
        run_frame();
      }
      game3d_present_trace::out.close();
      const auto isolation = where + "/isolation";
      continuity_log.where = isolation.c_str();
      isolate();
      require_records_match_log(where.c_str());
      script_name = nullptr;
      script_deadline = 0;
      continuity_log.where = "session";
      result.elapsed_ms = clock.wall_now() - started;
      const auto &judge = continuity_log.judge;
      result.rows = judge.rows();
      result.presents = judge.presents();
      result.legal = judge.legal_presents();
      result.shown = judge.shown();
      result.stalls = judge.stalls();
      result.frames = unsigned(value.frames.size());
      for (const auto &violation : continuity_log.judge.take()) {
        if (continuity::excused(violation) >= 0) {
          continue;
        }
        const auto found = present_frames.find(violation.present);
        const bool in_isolation = found == present_frames.end();
        const auto at = in_isolation ? unsigned(value.frames.size()) : found->second;
        const auto kind = std::string("continuity:") + continuity::name(violation.kind);
        const auto signature = kind + "/" + sunshine_game3d::name(violation.owner) + "/" + (violation.reason ? violation.reason : "");
        result.failures.push_back({kind, at, (in_isolation ? std::string("isolation") : frame_text(at)) + ": " + continuity::describe(violation), signature});
      }
      // unordered_read (the native provider fixture's require_ordered_reads):
      // a read the add-on recorded as ordered by the queue-order model that
      // the timeline's own Execute, Signal and Wait calls do not order.
      // A read the script's calls up to its Present order, but not with a
      // signal first submitted after it, is R1-late-signaller's (O1); any
      // other fails at the frame whose Present made it.
      result.ordered_reads = unsigned(script_order.reads.size());
      const auto late_row = std::size_t(std::find_if(std::begin(continuity::known_gaps), std::end(continuity::known_gaps), [](const continuity::gap_row &row) {
                                          return row.limit == continuity::limitation::late_signaller;
                                        }) -
                                        std::begin(continuity::known_gaps));
      for (const auto &read : script_order.judge_reads()) {
        const auto found = present_frames.find(read.present_index);
        const auto at = found == present_frames.end() ? unsigned(value.frames.size()) : found->second;
        if (read.late) {
          ++result.late_reads;
          ++result.rows[late_row].violations;
          continue;
        }
        result.failures.push_back({"unordered_read", at, (at < value.frames.size() ? frame_text(at) : std::string("isolation")) + ": " + read.text, "unordered_read"});
      }
      std::stable_sort(result.failures.begin(), result.failures.end(), [](const failure &a, const failure &b) {
        return a.frame < b.frame;
      });
      return result;
    }
  };

  // -- Shrinking -------------------------------------------------------------

  // Delta debugging (Zeller and Hildebrandt, "Simplifying and Isolating
  // Failure-Inducing Input", IEEE TSE 2002) over a failing timeline: a
  // candidate is kept when its run fails with the same signature, and is then
  // cut after the frame where it failed. Each candidate is a run, so the
  // cheapest reductions come first: each event kind back to its default in
  // every frame, then chunks of frames, then each event kind of each frame
  // of a short timeline. The budget bounds the runs.
  struct shrinker {
    timeline_fixture &fixture;
    std::string signature;
    unsigned budget {}, runs {};
    // No candidate starts that could not end by then (its frames' bound and
    // isolation).
    std::uint64_t stop_at {};

    // Whether frames fail with the signature; a kept candidate is cut after
    // the frame where it failed.
    bool fails(std::vector<generator::row> &frames) {
      if (runs >= budget || frames.empty() || fixture.clock.wall_now() + 60 * frames.size() + 3000 > stop_at) {
        return false;
      }
      ++runs;
      timeline_case candidate {0, frames};
      frame_events::normalize(candidate.frames);
      const auto result = fixture.run(candidate, "shrink-" + std::to_string(runs));
      const auto found = std::find_if(result.failures.begin(), result.failures.end(), [&](const failure &value) {
        return value.signature == signature;
      });
      std::printf("MEASURE session-timeline shrink run=%u frames=%zu %s\n", runs, frames.size(), found != result.failures.end() ? "reproduced" : "passed");
      if (found == result.failures.end()) {
        return false;
      }
      frames = candidate.frames;
      if (found->frame + 1 < frames.size()) {
        frames.resize(found->frame + 1);
      }
      return true;
    }

    std::vector<generator::row> shrink(std::vector<generator::row> current, unsigned failing_frame) {
      using namespace frame_events;
      // Nothing after the failing frame's Presents can matter.
      if (failing_frame + 1 < current.size()) {
        std::vector<generator::row> truncated(current.begin(), current.begin() + failing_frame + 1);
        if (fails(truncated)) {
          current = truncated;
        } else if (!fails(current)) {
          std::printf("MEASURE session-timeline shrink: the failure did not reproduce on replay; reporting the timeline unshrunk\n");
          return current;
        }
      }
      const auto reset = [](generator::row &value, unsigned d) {
        value[d] = 0;
        if (d == generated) {
          value[at] = 0;
        }
      };
      // Each event kind back to its default in every frame.
      for (const auto d : {sr_gpu, dlssg_gpu, hitch, order, ui, generated, drs, untag, dlssg, state, entry, sr, tag, fg}) {
        auto candidate = current;
        for (auto &value : candidate) {
          reset(value, d);
        }
        if (candidate != current && std::all_of(candidate.begin(), candidate.end(), legal) && fails(candidate)) {
          current = candidate;
        }
      }
      // ddmin over frames.
      for (std::size_t parts = 2; current.size() >= 2 && runs < budget;) {
        const auto chunk = (current.size() + parts - 1) / parts;
        bool removed = false;
        for (std::size_t begin = 0; begin < current.size() && runs < budget; begin += chunk) {
          auto candidate = current;
          candidate.erase(candidate.begin() + std::ptrdiff_t(begin), candidate.begin() + std::ptrdiff_t(std::min(current.size(), begin + chunk)));
          if (fails(candidate)) {
            current = candidate;
            parts = std::max<std::size_t>(parts - 1, 2);
            removed = true;
            break;
          }
        }
        if (!removed) {
          if (parts >= current.size()) {
            break;
          }
          parts = std::min(current.size(), parts * 2);
        }
      }
      // Each event kind of each frame, where few frames are left.
      for (std::size_t i = current.size(); i-- > 0 && current.size() <= 16 && runs < budget;) {
        for (unsigned d = 0; d != dimension_count && i < current.size() && runs < budget; ++d) {
          if (!current[i][d]) {
            continue;
          }
          auto candidate = current;
          reset(candidate[i], d);
          if (legal(candidate[i]) && fails(candidate)) {
            current = candidate;
          }
        }
      }
      normalize(current);
      return current;
    }
  };

  struct options {
    bool list {}, explicit_seed {}, explicit_events {}, trace {};
    std::uint64_t seed {}, seeds {fixed_timelines}, from {};
    unsigned shard {}, shards {1}, shrink_runs {40};
    std::string events, only;
  };

  bool parse_options(int argc, char **argv, options &out, std::vector<std::string> &positional, std::string &error) {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const bool value = i + 1 < argc;
      if (argument == "--list") {
        out.list = true;
      } else if (argument == "--trace") {
        out.trace = true;
      } else if (argument == "--timeline" && value) {
        out.explicit_seed = true;
        out.seed = std::strtoull(argv[++i], nullptr, 16);
      } else if (argument == "--events" && value) {
        out.explicit_events = true;
        out.events = argv[++i];
      } else if (argument == "--seeds" && value) {
        out.seeds = std::strtoull(argv[++i], nullptr, 10);
      } else if (argument == "--only" && value) {
        out.only = argv[++i];
      } else if (argument == "--from" && value) {
        out.from = std::strtoull(argv[++i], nullptr, 10);
      } else if (argument == "--shrink-runs" && value) {
        out.shrink_runs = unsigned(std::strtoul(argv[++i], nullptr, 10));
      } else if (argument == "--shard" && value) {
        char *end {};
        const auto text = argv[++i];
        out.shard = unsigned(std::strtoul(text, &end, 10));
        const bool separated = end && *end == '/';
        out.shards = separated ? unsigned(std::strtoul(end + 1, &end, 10)) : 0;
        if (!separated || !out.shards || out.shard >= out.shards || (end && *end)) {
          error = "--shard takes i/N with i < N";
          return false;
        }
      } else if (argument.rfind("--", 0) == 0) {
        error = "unknown argument '" + argument + "'";
        return false;
      } else {
        positional.push_back(argument);
      }
    }
    if (out.explicit_seed && out.explicit_events) {
      error = "--timeline and --events each select one timeline";
      return false;
    }
    if (!out.only.empty() && (out.explicit_seed || out.explicit_events)) {
      error = "--only selects a regression or witness, not a seed or an event list";
      return false;
    }
    return true;
  }

  std::string seed_text(std::uint64_t seed) {
    return seed ? generator::id_text(seed) : std::string("events");
  }

  std::string name_of(const timeline_case &value) {
    if (value.witness) {
      return std::string("witness=") + value.witness;
    }
    if (value.regression) {
      return std::string("regression=") + value.regression;
    }
    return "seed=" + seed_text(value.seed);
  }
}  // namespace

// The add-on lifecycle fixture (test_addon_unload_runtime.cpp) includes the
// timeline fixture alone.
#ifndef SUNSHINE_SESSION_TIMELINE_FIXTURE_ONLY
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  // --wall-clock: the real clock, never the game clock (test_game3d_clock.h).
  sunshine_game3d_fixture::take_wall_clock_option(argc, argv);
  using sunshine_game3d_fixture::game_clock;
  options chosen;
  std::vector<std::string> positional;
  std::string error;
  if (!parse_options(argc, argv, chosen, positional, error) || (!chosen.list && positional.size() != 5)) {
    std::fprintf(stderr, "%s%susage: reshade_game3d_session_timeline_runtime_test [--list] [--timeline <seed> | --events <list> | --seeds N [--from M]] [--only <regression or witness>] [--shard i/N] [--shrink-runs N] [--wall-clock] official.dll frozenShaders test.addon64 freshOutput interposer.dll\n", error.c_str(), error.empty() ? "" : "\n");
    return 2;
  }
  std::vector<timeline_case> selected_cases;
  if (chosen.explicit_seed) {
    selected_cases.push_back(generate(chosen.seed));
  } else if (chosen.explicit_events) {
    timeline_case value;
    if (!parse_events(chosen.events, value.frames, error)) {
      std::fprintf(stderr, "FAIL session timelines: %s\n", error.c_str());
      return 2;
    }
    selected_cases.push_back(value);
  } else {
    std::vector<timeline_case> cases;
    for (std::uint64_t i = 0; i != chosen.seeds; ++i) {
      cases.push_back(generate(seed_of(chosen.from + i)));
    }
    for (const auto &value : regressions) {
      timeline_case kept;
      kept.regression = value.fix;
      kept.lossless = value.lossless;
      kept.late_reads = value.late_reads;
      kept.ui_holds = value.ui_holds;
      kept.unidentified_holds = value.unidentified_holds;
      if (!parse_events(value.events, kept.frames, error)) {
        std::fprintf(stderr, "FAIL session timelines: the regression of %s: %s\n", value.fix, error.c_str());
        return 2;
      }
      cases.push_back(kept);
    }
    for (const auto &value : witnesses) {
      timeline_case witnessed;
      witnessed.witness = value.row;
      if (!parse_events(value.events, witnessed.frames, error)) {
        std::fprintf(stderr, "FAIL session timelines: the witness of %s: %s\n", value.row, error.c_str());
        return 2;
      }
      cases.push_back(witnessed);
    }
    for (std::size_t i = 0; i != cases.size(); ++i) {
      const auto &value = cases[i];
      const bool named = chosen.only.empty() || (value.regression && chosen.only == value.regression) ||
                         (value.witness && chosen.only == value.witness);
      if (i % chosen.shards == chosen.shard && named) {
        selected_cases.push_back(value);
      }
    }
    if (!chosen.only.empty() && selected_cases.empty()) {
      std::fprintf(stderr, "FAIL session timelines: --only %s names no regression or witness\n", chosen.only.c_str());
      return 2;
    }
  }
  const bool fixed_set = !chosen.explicit_seed && !chosen.explicit_events && chosen.from == 0 && chosen.seeds == fixed_timelines && chosen.shards == 1 && chosen.only.empty();
  std::size_t total_frames {};
  for (const auto &value : selected_cases) {
    total_frames += value.frames.size();
    if (chosen.list) {
      std::printf("timeline %s frames=%zu: %s\n", name_of(value).c_str(), value.frames.size(), describe(value.frames).c_str());
    }
  }
  if (chosen.list) {
    return 0;
  }
  // Each frame's bound and isolation, setup and the shrink budget; the
  // fixed set's own bound (wall time).
  const auto started = game_clock::wall_now();
  const auto budget_ms = fixed_set ? fixed_budget_ms : 60000 + 3000 * selected_cases.size() + 60 * total_frames + 6000 * std::uint64_t(chosen.shrink_runs);
  const auto watchdog_ms = DWORD(std::min<std::uint64_t>(budget_ms + 10000, 0x7fffffff));
  std::thread([watchdog_ms] {
    Sleep(watchdog_ms);
    std::fputs("FAIL session timelines watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    require(_putenv_s("SUNSHINE_DEPTH3D_EFFECT", "SunshineGame3D") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC", "1") == 0 && _putenv_s("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST", "1") == 0 && _putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", "0") == 0, "Could not configure the timeline runtime");
    width = 1280;
    height = 720;
    const auto directory = fs::absolute(positional[3]);
    require(!fs::exists(directory), "The timeline runtime requires a fresh isolated output directory");
    fs::create_directories(directory);
    const auto interposer = fs::absolute(positional[4]);
    require(fs::is_regular_file(interposer), "The 2.7.30 interposer DLL is missing");
    const auto isolated = directory / "sl.interposer.dll";
    fs::copy_file(interposer, isolated);
    const auto sdk = LoadLibraryW(isolated.c_str());
    require(sdk, "Could not load the isolated interposer");
    timeline_fixture fixture;
    fixture.runtime_directory = directory;
    fixture.initialize(fs::absolute(positional[0]), fs::absolute(positional[1]), directory, 2, 0, fs::absolute(positional[2]));
    fixture.load_native(directory);
    fixture.setup(sdk);
    fixture.trace = chosen.trace;
    // Setup is done: the timelines play game time on the test add-on's
    // clock, which engages its lead ahead of the real tick, so nothing setup
    // stamped survives into the first timeline (unless --wall-clock).
    require(fixture.clock.bind(GetModuleHandleW(L"SunshineSBSTest.addon64")), "The test add-on lacks the game clock");
    fixture.clock.engage();

    struct foreground_scope {
      timeline_fixture &fixture;

      ~foreground_scope() {
        fixture.set_foreground(nullptr);
      }
    } foreground {fixture};

    std::array<continuity::row_tally, continuity::gap_count> rows {};
    unsigned failed {}, presents {}, generated {}, legal {}, shown {}, frames_run {}, decisions {};
    // The first failing timeline (other than a closed witness), shrunk once
    // every timeline ran.
    const timeline_case *first_failing {};
    failure first_failure;
    for (std::size_t i = 0; i != selected_cases.size(); ++i) {
      const auto &value = selected_cases[i];
      auto label = "timeline-" + seed_text(value.seed);
      if (value.witness) {
        label = std::string("witness-") + value.witness;
      } else if (value.regression) {
        label = std::string("regression-") + value.regression;
      }
      auto result = fixture.run(value, label);
      if (value.witness) {
        std::size_t row = 0;
        while (row != continuity::gap_count && std::strcmp(continuity::known_gaps[row].id, value.witness)) {
          ++row;
        }
        require(row != continuity::gap_count, "A witness names no known gap row");
        if (!result.rows[row].violations) {
          result.failures.push_back({"witness:closed", 0, std::string("the witness of ") + value.witness + " lost no depth where its frames meet the limitation (" + std::to_string(result.rows[row].exercised) + " frames): the gap closed, so remove its row and witness", "witness:closed"});
        }
      }
      if (value.lossless) {
        // Losses only: a late read (unordered_read's R1 count) is no loss of
        // depth, and a timeline whose game signals late has some.
        unsigned recorded {};
        for (const auto &row : result.rows) {
          recorded += row.violations;
        }
        recorded -= result.late_reads;
        if (recorded) {
          result.failures.push_back({"regression:recorded_loss", 0, std::string("the regression of ") + value.regression + " lost depth where a known gap records it (" + std::to_string(recorded) + " losses): its fix keeps every Present", "regression:recorded_loss"});
        }
      }
      if (result.late_reads < value.late_reads) {
        result.failures.push_back({"regression:no_late_read", 0, std::string("the regression of ") + value.regression + " made " + std::to_string(result.late_reads) + " late reads of the " + std::to_string(value.late_reads) + " its scenario needs", "regression:no_late_read"});
      }
      if (result.ui_holds < value.ui_holds) {
        result.failures.push_back({"regression:no_ui_hold", 0, std::string("the regression of ") + value.regression + " judged " + std::to_string(result.ui_holds) + " UI generated-Present holds by depth's window of the " + std::to_string(value.ui_holds) + " its scenario needs", "regression:no_ui_hold"});
      }
      if (result.ui_unidentified_holds < value.unidentified_holds) {
        result.failures.push_back({"regression:no_unidentified_hold", 0, std::string("the regression of ") + value.regression + " made " + std::to_string(result.ui_unidentified_holds) + " UI holds at Presents without frame identity of the " + std::to_string(value.unidentified_holds) + " its scenario needs", "regression:no_unidentified_hold"});
      }
      for (std::size_t r = 0; r != rows.size(); ++r) {
        rows[r].exercised += result.rows[r].exercised;
        rows[r].violations += result.rows[r].violations;
      }
      presents += result.presents;
      generated += result.generated;
      legal += result.legal;
      shown += result.shown;
      frames_run += result.frames;
      decisions += result.decisions;
      std::printf("MEASURE session-timeline %s frames=%u presents=%u generated=%u legal=%u shown=%u stalls=%u decisions=%u refusals=%u preempted=%u cancels=%u ordered_reads=%u late_reads=%u failures=%zu elapsed_ms=%llu", name_of(value).c_str(), result.frames, result.presents, result.generated, result.legal, result.shown, result.stalls, result.decisions, result.refusals_matched, result.refusals_preempted, result.cancels, result.ordered_reads, result.late_reads, result.failures.size(), static_cast<unsigned long long>(result.elapsed_ms));
      for (std::size_t r = 0; r != rows.size(); ++r) {
        if (result.rows[r].exercised) {
          std::printf(" %s={supplied=%u violations=%u}", continuity::known_gaps[r].id, result.rows[r].exercised, result.rows[r].violations);
        }
      }
      std::printf("\n");
      if (result.failures.empty()) {
        continue;
      }
      ++failed;
      for (std::size_t f = 0; f != result.failures.size() && f != 8; ++f) {
        std::printf("MEASURE session-timeline failure %s %s: %s\n", name_of(value).c_str(), result.failures[f].kind.c_str(), result.failures[f].text.c_str());
      }
      const auto &first = result.failures.front();
      std::printf("FAIL timeline %s frames=%zu %s at frame %u\nEVENTS %s: %s\n", name_of(value).c_str(), value.frames.size(), first.kind.c_str(), first.frame, name_of(value).c_str(), describe(value.frames).c_str());
      for (auto f = first.frame > 8 ? first.frame - 8 : 0u; f <= first.frame && f < value.frames.size(); ++f) {
        std::printf("CONTEXT %s frame %u: %s\n", name_of(value).c_str(), f, generator::describe(frame_events::family(), value.frames[f]).c_str());
      }
      if (!first_failing && first.kind != "witness:closed" && first.kind != "regression:recorded_loss") {
        first_failing = &value;
        first_failure = first;
      }
    }
    if (first_failing && chosen.shrink_runs) {
      // Within the budget, and in the fixed set (ctest) for no longer than
      // shrink_ms: the reproducer is printed well before ctest's timeout.
      const auto shrink_from = game_clock::wall_now();
      const auto stop_at = fixed_set ? std::min(started + budget_ms, shrink_from + shrink_ms) : started + budget_ms;
      shrinker reduce {fixture, first_failure.signature, chosen.shrink_runs, 0, stop_at};
      const auto minimal = reduce.shrink(first_failing->frames, first_failure.frame);
      // The reproducer once more, after every candidate: in this process each
      // run follows the earlier ones, and a fresh process's --events replay
      // may still differ in timing.
      reduce.budget = reduce.runs + 1;
      reduce.stop_at = std::max(reduce.stop_at, game_clock::wall_now() + 60 * minimal.size() + 3000);
      auto replay = minimal;
      const bool reproduced = reduce.fails(replay);
      std::printf("REPRODUCER %s %s runs=%u frames=%zu reproduced=%s: --events \"%s\"\n", name_of(*first_failing).c_str(), first_failure.signature.c_str(), reduce.runs, minimal.size(), reproduced ? "yes" : "no", describe(minimal).c_str());
    }
    // The game clock's barriers over the run (their times are wall time).
    fixture.clock.report("session timelines");
    std::printf("MEASURE session-timelines timelines=%zu frames=%u presents=%u generated=%u legal=%u shown=%u decisions=%u failed=%u", selected_cases.size(), frames_run, presents, generated, legal, shown, decisions, failed);
    for (std::size_t r = 0; r != rows.size(); ++r) {
      std::printf(" %s={supplied=%u violations=%u}", continuity::known_gaps[r].id, rows[r].exercised, rows[r].violations);
    }
    std::printf("\nMEASURE session-timelines found-defects %s\n", continuity::found_defects(rows).c_str());
    reshade::unregister_event<reshade::addon_event::reshade_present>(record_present);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
    if (failed) {
      std::fprintf(stderr, "FAIL session timelines: %u of %zu timelines failed (seeds and reproducers above)\n", failed, selected_cases.size());
      return 1;
    }
    std::printf("PASS session timelines: %zu seeded timelines (%u frames, %u Presents) judged by the continuity invariant and the oracle at each copy decision\n", selected_cases.size(), frames_run, presents);
    return 0;
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "FAIL session timelines: %s\n", failure.what());
    return 1;
  }
}
#endif
