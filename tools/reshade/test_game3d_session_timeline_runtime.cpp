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
//   while that evaluation copies its own depth input (2x to 4x);
// - a dynamic-resolution step of the tagged extent;
// - the viewport's SR evaluation with its own depth input on the presenting
//   queue or the game's own direct queue, or of the frame's tag;
// - DLSS-G's own evaluation on a direct or compute list, its depth in a state
//   recorded there or not;
// - the game's HUD-less and Backbuffer tags, with or without a command list,
//   or untagged in the same frame.
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
// Command line: [--list] [--timeline <seed> | --events <list> | --seeds N
// [--from M]] [--shard i/N] [--shrink-runs N] [--trace] official.dll
// frozenShaders test.addon64 freshOutput interposer.dll. Without a selection
// it runs the fixed set, the first fixed_timelines seeds, the regressions
// and the witnesses, as the registered test does; --trace prints every
// Present's frame record.
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

namespace {
  namespace matrix = game3d_contract;
  namespace generator = game3d_contract::generator;
  namespace refusals = sunshine_depth_refusal;
  namespace continuity = game3d_continuity;

  // The fixed set the registered test runs, and each timeline's length.
  constexpr unsigned fixed_timelines = 24, min_frames = 60, max_frames = 200;
  // The fixed set's time bound, below ctest's 240 s, and how long its first
  // failure is shrunk once every timeline ran: a failing run holds the GPU
  // about a minute longer than a passing one and prints its reproducer in
  // time. --shrink-runs outside the fixed set shrinks for longer.
  constexpr std::uint64_t fixed_budget_ms = 225000, shrink_ms = 60000;

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
      dimension_count
    };

    enum class fg_event {
      keep,
      on,
      off
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
      other_queue  // Its own input on the game's own direct queue (known gap C1).
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
          {"fg", {"keep", "on", "off"}},
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
        },
        legal,
      };
      return value;
    }

    // Relative weights of each dimension's values, and the chance in a
    // thousand that a frame keeps the previous frame's value. FG and dynamic
    // resolution change by one-time events; FG starts off.
    const generator::draw &drawing() {
      static const generator::draw value {
        {
          {940, 30, 30},
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
        if (!legal(value)) {
          value[state] = 0;
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
  };

  // Each Assassin's Creed Shadows fix has a regression timeline: an event
  // list shaped like the session it fixed, which the fixed set runs after
  // its seeds and judges as any timeline. Each fails with its fix reverted
  // (README, Session-level tests); random seeds meet these shapes only by
  // chance, and a change of the event language changes every seed. So does
  // each hold the session tests' review found unjudged by a mutation.
  struct regression {
    const char *fix, *events;
    // A loss a known gap records fails it too: with its fix reverted, the
    // Presents it loses meet a row's limitation (C6's: C1, its DLSS-G compute
    // copy readable three evaluations later), and with the fix it loses none.
    bool lossless {};
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
    {"C8-fg-off-sr-other-queue", "20*sr=local; fg=on,tag=present_list; fg=off,sr=other_queue; 4*sr=other_queue; fg=on,sr=other_queue; 15*sr=other_queue"},
    // FG Off releases the queue's API association, yet the FG copy is held
    // within the source age (known gap C8 until then: color-only Presents
    // after FG Off dropped it and Generic's decoy view took the queue, so
    // the SR copy on the game's own queue that followed within that age
    // found nothing to hold): one FG tag copy, FG off with frames that
    // supply nothing, then the viewport's SR input on the game's queue.
    {"C8-fg-off-color-only", "20*sr=local; fg=on,tag=present_list; fg=off; 10*defaults; 15*sr=other_queue"},
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
    {"fg-on-holds-sr-copy", "20*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 5*tag=now_list,sr=local,generated=1; fg=off,sr=local; 10*sr=local; fg=on,tag=now_list,generated=1,at=inside; 5*tag=now_list,generated=1; fg=off,sr=local; 10*sr=local; fg=on,tag=now_list,sr=local,generated=1,at=evaluate; 5*sr=local,generated=1"},
  };

  // Each known gap the timelines found has a witness: an event list whose
  // frames meet its limitation alone, long enough that every run loses depth
  // there. The fixed set runs them after its seeds, and a witness that loses
  // none fails it (the gap closed: its fix removes the row and the witness,
  // which becomes a regression timeline), as the matrix's gap seeds keep a
  // recorded gap's own failing cell. None is open: the last, C5's, closed on
  // 10-26.
  struct witness {
    const char *row, *events;
  };

  constexpr std::array<witness, 0> witnesses {};

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
      out.frames.push_back(generator::sample(events::family(), rng, events::drawing(), out.frames.empty() ? nullptr : &out.frames.back()));
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
    // The add-on's totals of depth copies deferred to their frame's Present,
    // of those copied and of the deferrals an untag cancelled.
    void (*deferral_totals)(std::uint64_t *, std::uint64_t *, std::uint64_t *) {};
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
    // Whether the frame's tag copy is deferred to its Present, and whether
    // it waits there for DLSS-G to evaluate its frame
    // (frame_generation::deferral_evaluations): DLSS-G evaluated a frame
    // since the previous frame whose tag copy was deferred, or none was.
    bool deferred_tag {}, awaits_dlssg {};
    // When DLSS-G last evaluated a frame (copied, deferred to a tag or
    // refused), the frames it evaluated, and their count at the previous
    // frame whose tag copy was deferred (deferred_known: one was).
    std::uint64_t dlssg_evaluated {}, dlssg_frames {}, deferred_dlssg_frames {};
    bool deferred_known {};
    // Whether the previous frame supplied an FG copy, so that a live FG
    // scope owns this one (continuity::supply::owned).
    bool fg_supplied {};
    std::map<std::uint64_t, unsigned> present_frames;

    // What the frame being recorded supplies (the continuity supply log).
    struct frame_supply {
      std::uint8_t legal {}, limited {}, permitted {};
      continuity::limitation limit {continuity::limitation::none};
      std::uint64_t tick {};
    } supply;

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
      deferral_totals = reinterpret_cast<decltype(deferral_totals)>(GetProcAddress(addon, "SunshineStreamlineTestDeferrals"));
      require(deferral_totals, "Test add-on lacks its deferral totals");
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
      // The native observer installs its hooks for the new queues before any
      // timeline is judged.
      render_tracked_depth = {};
      for (const auto until = GetTickCount64() + sunshine_freshness::association_ms; GetTickCount64() < until;) {
        present_frame();
      }
    }

    void create_queues() {
      game_work.create(game.p, D3D12_COMMAND_LIST_TYPE_DIRECT);
      dlssg_work.create(game.p, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    }

    // Retires every ring recording (as a game does at its end), once their
    // copies are older than the source age.
    void drain() {
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
      supply.tick = supply.tick ? supply.tick : GetTickCount64();
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
          const bool preempted = !matched && !contract && (!seen.empty() || frame_generation_scope());
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

      struct end_generated {
        ~end_generated() {
          continuity_log.generated = false;
        }
      } scope;

      if (inside) {
        checked(swapchain->Present(0, 0), "Generated Present inside the recording");
      } else {
        render_tracked_depth = {};
        present_frame();
      }
      ++result.generated;
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
    // (S9 and R7, an untag that cancels the deferred copy) supply none.
    void judge_shown(const char *which) {
      const auto s = inspect().source;
      if (trace) {
        trace_present(which, s);
      }
      if (!(s.selected && s.ready)) {
        return;
      }
      const bool fresh = !s.reused_depth && s.current.capture && s.current.capture != last_capture;
      const auto now = GetTickCount64();
      const auto within = [now](std::uint64_t tick, std::uint64_t slack) {
        return tick && now - tick < sunshine_freshness::source_age_ms + slack;
      };
      if (fresh && !other_queue_pending()) {
        if (!within(api_supplied, continuity::tick_slack_ms)) {
          fail("oracle:unsupplied_fresh", frame_text(frame) + ": " + which + " showed a fresh copy (capture " + std::to_string(s.current.capture) + ") with no API source supplied within the source age", frame, s.frame_generation_input ? "fg_input" : "scene");
        } else if (s.frame_generation_input && !within(fg_copy, continuity::tag_slack_ms)) {
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
      std::printf("TRACE frame=%u %s present=%llu tick=%llu owner=%s state=%s reason=%s fg=%s/%s scene=%s/%s legal=0x%x limited=0x%x permitted=0x%x shown=%u capture=%llu reused=%u fg_input=%u\n", frame, which, static_cast<unsigned long long>(r.present), static_cast<unsigned long long>(r.tick_ms), sunshine_game3d::name(r.depth_owner), sunshine_game3d::name(r[owner].state), r[owner].reason, sunshine_game3d::name(r[sunshine_game3d::source_slot::fg_depth].state), r[sunshine_game3d::source_slot::fg_depth].reason, sunshine_game3d::name(r[sunshine_game3d::source_slot::scene_depth].state), r[sunshine_game3d::source_slot::scene_depth].reason, unsigned(supply.legal), unsigned(supply.limited), unsigned(supply.permitted), unsigned(s.selected && s.ready), static_cast<unsigned long long>(s.current.capture), unsigned(s.reused_depth), unsigned(s.frame_generation_input));
    }

    bool other_queue_pending() const {
      return other_queue_copy && GetTickCount64() - other_queue_copy < sunshine_freshness::source_age_ms + continuity::tag_slack_ms;
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
      deferred_tag = awaits_dlssg = false;
      draw(*decoy);
      supply.permitted |= continuity::generic_boundary;
      supply.tick = GetTickCount64();
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
        if (deferred_tag) {
          awaits_dlssg = !deferred_known || deferred_dlssg_frames != dlssg_frames;
          deferred_known = true;
          deferred_dlssg_frames = dlssg_frames;
        }
        bool supplied = false;
        if (call.legal && requires_capture(call.expected)) {
          offer(call, continuity::tag_copy);
          supplied = true;
        }
        if (sr_choice == sr_form::tag) {
          const auto evaluated = tag_case(matrix::copy_point::same_frame_evaluate, generated_inside);
          decision at_evaluate {"tag_at_evaluate", tag_outcome(evaluated), matrix::legal(evaluated), false, {"evaluate"}};
          decisions.push_back(at_evaluate);
          if (!supplied && at_evaluate.legal && requires_capture(at_evaluate.expected)) {
            offer(at_evaluate, continuity::sr_evaluate);
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
        returned(frame_entry ? sl.frame_tag(*token, view, &depth_tag, 1, tag_commands) : sl.global_tag(view, &depth_tag, 1, tag_commands), "the depth tag call");
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
        returned(sl.evaluate(0, *token, inputs, sr_choice == sr_form::local ? 2 : 1, command()), "slEvaluateFeature");
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
        decide("sr_other_queue", local_case(declared, false, 0), continuity::sr_evaluate, {"evaluate"}, false, continuity::limitation::cross_queue_sr);
        returned(sl.evaluate(0, *token, inputs, 2, reinterpret_cast<void *>(native)), "slEvaluateFeature on the game's queue");
        poll("evaluate");
        game_work.execute(n, queue.p);
        other_queue_copy = GetTickCount64();
      }
      dlssg_evaluation(of<dlssg_form>(value, events::dlssg));
      move(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      // A frame whose deferred FG tag copy waits for DLSS-G to evaluate its
      // frame, which DLSS-G does not do, supplies that copy as permitted
      // only: DLSS-G evaluated the previous deferred frame, so the add-on
      // cannot tell this frame's Present from an earlier frame's, and the
      // Present holds the earlier copy. Any later frame without DLSS-G's
      // evaluation copies at its Present (known gap C7 until 10-26, when
      // each waited within the source age of any evaluation). A frame whose
      // only FG input is DLSS-G's compute copy supplies its live FG scope,
      // which keeps the viewport's SR input out while that copy is readable
      // only frames later (known gap C1). DLSS-G's evaluation of a frame
      // without an FG tag of its own copies (known gap C6 until 10-26, when
      // it deferred to an earlier frame's tag copy for the source age).
      const auto dlssg_choice = of<dlssg_form>(value, events::dlssg);
      const auto now = GetTickCount64();
      if (fg_on && deferred_tag && awaits_dlssg && dlssg_choice == dlssg_form::none && dlssg_evaluated && now - dlssg_evaluated < sunshine_freshness::source_age_ms + continuity::tag_slack_ms && (supply.legal & continuity::tag_copy)) {
        supply.legal &= std::uint8_t(~continuity::tag_copy);
        supply.permitted |= continuity::tag_copy;
      } else if (fg_on && dlssg_choice == dlssg_form::compute && fg_copy && now - fg_copy < sunshine_freshness::fg_silent_ms + continuity::tag_slack_ms && !(supply.legal & continuity::tag_copy) && (supply.legal & continuity::sr_evaluate)) {
        supply.legal &= std::uint8_t(~continuity::sr_evaluate);
        supply.limited |= continuity::sr_evaluate;
        supply.limit = continuity::limitation::cross_queue_sr;
      }
      // A live FG scope keeps the viewport's SR input out of the frame after
      // one that supplied an FG copy, which it expects to supply too: the
      // Present shows the scope's FG depth instead (FG keeps the frames it
      // supplies; the next frame without an FG copy turns the scope silent).
      const bool owned = std::exchange(fg_supplied, false) && fg_on;
      // An FG copy its scope accepts: an FG tag's, or DLSS-G's in a state
      // its list recorded (its compute copy is accepted later, with its own
      // tick).
      if (fg_on && frame_generation_scope() && ((supply.legal | supply.permitted | supply.limited) & (continuity::tag_copy | continuity::dlssg_input))) {
        fg_copy = now;
        fg_supplied = true;
      }
      if (fg_on && dlssg_choice != dlssg_form::none) {
        dlssg_evaluated = now;
        ++dlssg_frames;
      }
      // The frame's supply: a frame whose legal sources all meet a known
      // limitation is tagged with it; any other legal source must show.
      continuity_log.pending = {};
      continuity_log.pending.legal = std::uint8_t(supply.legal | supply.limited);
      continuity_log.pending.permitted = supply.permitted;
      continuity_log.pending.limited = supply.limited;
      continuity_log.pending.owned = owned ? std::uint8_t(supply.legal & continuity::sr_evaluate) : std::uint8_t {};
      continuity_log.pending.limit = supply.legal ? continuity::limitation::none : supply.limit;
      continuity_log.pending.tick = supply.tick;
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
        // presenting queue reads its copy once DLSS-G reset that ring entry.
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
        decide("dlssg_compute", value, continuity::dlssg_input, {"dlssg"}, true, continuity::limitation::cross_queue_sr);
        require(ngx_evaluate(compute, feature, &dlssg_inputs, nullptr) == ngx_fixture::success, "DLSS-G's compute evaluation changed the SDK result");
        poll("dlssg");
        if (observed) {
          transition(compute, selected->resource.p, read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        dlssg_work.execute(n);
        if (observed) {
          other_queue_copy = GetTickCount64();
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
      require(ngx_evaluate(list, feature, &dlssg_inputs, nullptr) == ngx_fixture::success, "DLSS-G's evaluation changed the SDK result");
      poll("dlssg");
      if (observed) {
        transition(list, hudless.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        transition(list, backbuffer.p, read, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }
    }

    void set_frame_generation(bool on) {
      const fg_options value {{nullptr, fg_options_type, 3}, on ? 1u : 0u, on ? 1u : 0u};
      returned(sl.set_options(view, value), "slDLSSGSetOptions");
      fg_on = on;
    }

    void run_frame() {
      using namespace frame_events;
      const auto &value = (*frames)[frame];
      const auto event = of<fg_event>(value, events::fg);
      if (event != fg_event::keep) {
        set_frame_generation(event == fg_event::on);
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
      };
      present_frame();
      render_tracked_depth = {};
      poll("present");
      if (!record_history.records.empty()) {
        present_frames[record_history.records.back().present] = frame;
      }
      if (supply.legal || supply.limited || (supply.permitted & continuity::api_sources)) {
        api_supplied = GetTickCount64();
      }
      judge_decisions();
      if (trace) {
        for (const auto &value : polled) {
          std::printf("TRACE frame=%u refusal %s at %s adapter=%u viewport=%u class=%u gate=%u stage=%u fg=%u\n", frame, value.value.name, value.step, unsigned(value.value.provider), value.value.viewport, unsigned(value.value.reason), unsigned(value.value.where), unsigned(value.value.stage), unsigned(value.value.frame_generation));
        }
      }
      judge_shown("its Present");
      for (unsigned i = 0, n = unsigned(value[events::generated]); i != n && of<generated_at>(value, events::at) == generated_at::between; ++i) {
        generated_present(false);
      }
    }

    // Retires the run's viewport (FG off, the depth untagged, retired_viewports
    // scratch viewports' constants), presents past the association with no SDK call
    // or draw, then retires the rings' recordings and resets calibration.
    void isolate() {
      using namespace sunshine_streamline;
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
      for (const auto until = GetTickCount64() + sunshine_freshness::association_ms + 200; GetTickCount64() < until;) {
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
      present_frames.clear();
      where = label;
      continuity_log.judge = {};
      continuity_log.pending = {};
      continuity_log.where = where.c_str();
      script_name = where.c_str();
      script_deadline = GetTickCount64() + 15000 + 150 * value.frames.size();
    }

    run_result run(const timeline_case &value, const std::string &label) {
      const auto started = GetTickCount64();
      begin_run(value, label, next_viewport++);
      fg_on = false;
      for (frame = 0; frame != value.frames.size(); ++frame) {
        run_frame();
      }
      const auto isolation = where + "/isolation";
      continuity_log.where = isolation.c_str();
      isolate();
      require_records_match_log(where.c_str());
      script_name = nullptr;
      script_deadline = 0;
      continuity_log.where = "session";
      result.elapsed_ms = GetTickCount64() - started;
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
      if (runs >= budget || frames.empty() || GetTickCount64() + 60 * frames.size() + 3000 > stop_at) {
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
      for (const auto d : {ui, generated, drs, untag, dlssg, state, entry, sr, tag, fg}) {
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
    std::string events;
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
  options chosen;
  std::vector<std::string> positional;
  std::string error;
  if (!parse_options(argc, argv, chosen, positional, error) || (!chosen.list && positional.size() != 5)) {
    std::fprintf(stderr, "%s%susage: reshade_game3d_session_timeline_runtime_test [--list] [--timeline <seed> | --events <list> | --seeds N [--from M]] [--shard i/N] [--shrink-runs N] official.dll frozenShaders test.addon64 freshOutput interposer.dll\n", error.c_str(), error.empty() ? "" : "\n");
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
      if (i % chosen.shards == chosen.shard) {
        selected_cases.push_back(cases[i]);
      }
    }
  }
  const bool fixed_set = !chosen.explicit_seed && !chosen.explicit_events && chosen.from == 0 && chosen.seeds == fixed_timelines && chosen.shards == 1;
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
  // fixed set's own bound.
  const auto started = GetTickCount64();
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
        unsigned recorded {};
        for (const auto &row : result.rows) {
          recorded += row.violations;
        }
        if (recorded) {
          result.failures.push_back({"regression:recorded_loss", 0, std::string("the regression of ") + value.regression + " lost depth where a known gap records it (" + std::to_string(recorded) + " losses): its fix keeps every Present", "regression:recorded_loss"});
        }
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
      std::printf("MEASURE session-timeline %s frames=%u presents=%u generated=%u legal=%u shown=%u stalls=%u decisions=%u refusals=%u preempted=%u cancels=%u failures=%zu elapsed_ms=%llu", name_of(value).c_str(), result.frames, result.presents, result.generated, result.legal, result.shown, result.stalls, result.decisions, result.refusals_matched, result.refusals_preempted, result.cancels, result.failures.size(), static_cast<unsigned long long>(result.elapsed_ms));
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
      const auto shrink_from = GetTickCount64();
      const auto stop_at = fixed_set ? std::min(started + budget_ms, shrink_from + shrink_ms) : started + budget_ms;
      shrinker reduce {fixture, first_failure.signature, chosen.shrink_runs, 0, stop_at};
      const auto minimal = reduce.shrink(first_failing->frames, first_failure.frame);
      // The reproducer once more, after every candidate: in this process each
      // run follows the earlier ones, and a fresh process's --events replay
      // may still differ in timing.
      reduce.budget = reduce.runs + 1;
      reduce.stop_at = std::max(reduce.stop_at, GetTickCount64() + 60 * minimal.size() + 3000);
      auto replay = minimal;
      const bool reproduced = reduce.fails(replay);
      std::printf("REPRODUCER %s %s runs=%u frames=%zu reproduced=%s: --events \"%s\"\n", name_of(*first_failing).c_str(), first_failure.signature.c_str(), reduce.runs, minimal.size(), reproduced ? "yes" : "no", describe(minimal).c_str());
    }
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
