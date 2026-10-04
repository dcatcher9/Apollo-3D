// SPDX-License-Identifier: GPL-3.0-only
// Rule H2 (UI framework M5, fix 2; game3d_still_screen.h): the SDR scope, the
// predicate's edges, the run (entry exactly at run_ms, release on the first
// failing sample, ends by scope, identity and unmeasured renders, short runs,
// the sample gap bound before entry), the guard's
// decode of texel 12 and the reused bit, and two live streams: Stellar
// Blade SDR's loading screen (sb_s2b_0719.log 07:19:57-07:20:06, about 12%
// of its samples on reused depth) and Expedition 33's SDR FG-off gameplay dips
// (e33_s2b.log 07:43:02-07:43:23).
#include "game3d_scene_guard.h"
#include "game3d_still_screen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace sunshine_game3d;
  namespace still = ui_detection::still;
  using still_screen::cells;
  using still_screen::end_reason;

  void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
  }

  // A measured in-scope sample without a decided source: n edge cells, D,
  // and still cells of all cells compared.
  still_screen::input sample(std::uint64_t tick, float d, std::uint32_t still_cells = cells, std::uint32_t n = 400) {
    return {tick, true, 0u, false, true, n, d, still_cells, cells};
  }
  // The still cells of a share in percent, rounded up so that 95 is exactly
  // the bound.
  constexpr std::uint32_t share(std::uint32_t percent) { return (cells * percent + 99u) / 100u; }

  void scope_is_sdr_output() {
    for (const std::uint32_t format : {24u, 87u, 28u, 91u, 23u, 88u, 29u})
      require(still_screen::sdr_output(1, format), "An 8/10-bit SDR swapchain format is out of scope: " + std::to_string(format));
    require(!still_screen::sdr_output(3, 24) && !still_screen::sdr_output(2, 10) && !still_screen::sdr_output(4, 24) &&
        !still_screen::sdr_output(1, 10) && !still_screen::sdr_output(1, 11) && !still_screen::sdr_output(1, 0) &&
        !still_screen::sdr_output(0, 24), "PQ, scRGB, HLG, a float or 16-bit format, or an unknown space is in scope");
    std::puts("PASS H2 scope: sRGB colour space with 8/10-bit UNORM or 8-bit *_SRGB only; PQ, scRGB, HLG and float never");
  }

  void predicate_edges() {
    require(still_screen::passes(sample(0, .05f)) && !still_screen::passes(sample(0, .0501f)) &&
        still_screen::failure(sample(0, .0501f)) == end_reason::not_hidden, "D 0.05 must pass and 0.0501 fail (not_hidden)");
    require(still_screen::passes(sample(0, -.068f)) && still_screen::passes(sample(0, -1.f)),
      "A negative D (hidden) must pass");
    require(still_screen::passes(sample(0, 0.f, cells, ui_detection::scene::min_edges)) &&
        still_screen::failure(sample(0, 0.f, cells, ui_detection::scene::min_edges - 1)) == end_reason::unmeasured,
      "n 128 must pass and 127 be unmeasured");
    auto partial = sample(0, 0.f);
    partial.compared = cells - 1;
    partial.still = cells - 1;
    require(still_screen::failure(partial) == end_reason::unmeasured, "A sample with an uncompared cell must be unmeasured");
    require(still_screen::passes(sample(0, 0.f, share(95))) &&
        still_screen::failure(sample(0, 0.f, share(95) - 1)) == end_reason::moving,
      "A 95% still share must pass and 94.99% be moving");
    require(std::uint64_t(share(95) - 1) * 100u < 95u * cells && std::uint64_t(share(95)) * 100u >= 95u * cells,
      "The share bound is not exactly 95%");
    for (const std::uint32_t source : {0u, ui_detection::source_still}) {
      auto s = sample(0, 0.f);
      s.source = source;
      require(still_screen::passes(s), "Source 0 or 11 must not count as decided");
    }
    for (const std::uint32_t source : {1u, 4u, 6u, 8u, 10u}) {
      auto s = sample(0, 0.f);
      s.source = source;
      require(still_screen::failure(s) == end_reason::source, "A decided source must fail the predicate");
    }
    auto reused = sample(0, 0.f);
    reused.reused = true;
    require(still_screen::failure(reused) == end_reason::source, "A T1-reused decision must count as decided");
    auto out = sample(0, 0.f);
    out.in_scope = false;
    require(still_screen::failure(out) == end_reason::scope, "A sample out of scope must fail as scope");
    auto unran = sample(0, 0.f);
    unran.ran = false;
    require(still_screen::failure(unran) == end_reason::unmeasured, "A sample without evidence must be unmeasured");
    require(still_screen::failure(sample(0, 0.f, 0, 400)) == end_reason::moving, "A sample with no still cell must be moving");
    std::puts("PASS H2 predicate: D <= 0.05, n >= 128, every cell compared and 95% still, no decided or reused source, in scope");
  }

  void run_enters_and_releases() {
    still_screen::run r;
    require(r.state() == still_screen::phase::none && !r.run_ms() && r.wants_measure(), "A new run is not empty");
    // Samples every 100 ms: entered exactly when the run spans 2000 ms.
    for (std::uint64_t tick = 1000; tick < 3000; tick += 100) {
      const auto o = r.observe(sample(tick, -.02f));
      require(!o.entered && !o.ended && r.state() == still_screen::phase::pending, "The run entered before 2000 ms");
    }
    require(r.run_ms() == 1900, "The pending run does not report its length");
    auto o = r.observe(sample(3000, -.03f, share(97)));
    require(o.entered && r.active() && r.state() == still_screen::phase::active && r.run_ms() == still::run_ms,
      "The run did not enter at exactly 2000 ms");
    const auto &current = r.current();
    require(current.samples == 21 && current.d_min == -.03f && current.d_max == -.02f &&
        current.share_min == float(share(97)) / float(cells), "The episode's D range, share or samples are wrong");
    o = r.observe(sample(3100, -.02f));
    require(!o.entered && r.active(), "An active run re-entered");
    // The first failing sample releases it at once.
    o = r.observe(sample(3200, -.02f, share(90)));
    require(o.ended && o.reason == end_reason::moving && !r.active() && r.state() == still_screen::phase::none &&
        o.ended_episode.first == 1000 && o.ended_episode.last == 3100 && o.ended_episode.samples == 22 && !o.short_run,
      "The first moving sample did not release the active run");
    // A run shorter than 2000 ms ends as a short run with its length.
    r.observe(sample(4000, 0.f));
    r.observe(sample(4500, 0.f));
    o = r.observe(sample(4600, .3f));
    require(!o.ended && o.short_run && o.short_ms == 500 && o.reason == end_reason::not_hidden, "A short run was not reported");
    // A failing sample without a run reports nothing.
    o = r.observe(sample(4700, .3f));
    require(!o.ended && !o.short_run, "A failing sample without a run reported an end");
    // A tick before the last restarts the run.
    r.observe(sample(5000, 0.f));
    r.observe(sample(5100, 0.f));
    o = r.observe(sample(5050, 0.f));
    require(o.short_run && o.short_ms == 100 && r.state() == still_screen::phase::pending && !r.run_ms(),
      "A tick going backwards did not restart the run");
    // wants_measure: false only after a sample that decided a source other
    // than 11.
    auto decided = sample(5200, 0.f);
    decided.source = 4;
    r.observe(decided);
    require(!r.wants_measure(), "A decided source must stop H2's own measurement");
    decided.source = ui_detection::source_still;
    r.observe(decided);
    require(r.wants_measure(), "A flattened sample (11) must keep measuring");
    r.observe(sample(5300, 0.f, 0, 0));
    require(r.wants_measure(), "An unmeasured sample without a source must keep measuring");
    std::puts("PASS H2 run: enters exactly at 2000 ms of passing samples, releases on the first failing one, reports short runs");
  }

  void leave_and_identity() {
    still_screen::run r;
    for (std::uint64_t tick = 0; tick <= 2500; tick += 100) r.observe(sample(tick, 0.f));
    auto o = r.leave(end_reason::scope);
    require(o.ended && o.reason == end_reason::scope && o.ended_episode.duration_ms() == 2500 && !r.active(),
      "Leaving scope did not end the active run");
    for (std::uint64_t tick = 0; tick <= 700; tick += 100) r.observe(sample(tick, 0.f));
    o = r.leave(end_reason::scope);
    require(!o.ended && o.short_run && o.short_ms == 700, "Leaving scope did not end a pending run as short");
    require(!r.leave(end_reason::scope).short_run, "Leaving without a run reported one");
    // The guard: only an identity change clears the run, and says what it
    // ended.
    scene_guard::state guard;
    require(!guard.enter_scope(1, 7).ended, "The first scope ended a run");
    for (std::uint64_t tick = 0; tick <= 2000; tick += 100) guard.still.observe(sample(tick, 0.f));
    require(guard.still_flatten() && !guard.enter_scope(1, 7).ended && guard.still_flatten(),
      "The same identity cleared the run");
    o = guard.enter_scope(2, 7);
    require(o.ended && o.reason == end_reason::identity && !guard.still_flatten() &&
        guard.still.state() == still_screen::phase::none, "An epoch change did not end the run as identity");
    for (std::uint64_t tick = 0; tick <= 300; tick += 100) guard.still.observe(sample(tick, 0.f));
    o = guard.enter_scope(2, 8);
    require(!o.ended && o.short_run && o.short_ms == 300 && o.reason == end_reason::identity,
      "A viewport change did not end a pending run as identity");
    // A render in scope that cannot be sampled ends the run (unmeasured)
    // but keeps what the latest sample decided for wants_measure.
    still_screen::run u;
    for (std::uint64_t tick = 0; tick <= 2000; tick += 100) u.observe(sample(tick, 0.f));
    o = u.leave(end_reason::unmeasured);
    require(o.ended && o.reason == end_reason::unmeasured && !u.active(), "An unmeasured render did not end the active run");
    auto decided = sample(2100, 0.f);
    decided.source = 4;
    u.observe(decided);
    require(!u.wants_measure() && !u.leave(end_reason::unmeasured).short_run && !u.wants_measure() &&
        (u.leave(end_reason::scope), u.wants_measure()), "An unmeasured end changed wants_measure, or a scope end kept it");
    std::puts("PASS H2 ends: scope, identity and unmeasured renders end a run at once; only an identity change clears the "
      "guard's run");
  }

  // The 2 s rests on samples about 100 ms apart: a run that has not entered
  // restarts at a sample more than max_gap_ms after its last one, so two
  // passing samples around a stall never enter; an active run is released
  // only by a failing sample.
  void run_needs_dense_samples() {
    still_screen::run r;
    r.observe(sample(1000, .02f));
    auto o = r.observe(sample(3500, .02f));
    require(!o.entered && !r.active() && o.short_run && o.short_ms == 0 && o.reason == end_reason::unmeasured &&
        r.state() == still_screen::phase::pending && r.current().first == 3500, "Two samples 2.5 s apart entered");
    // A gap of exactly max_gap_ms keeps the run; one past it restarts it.
    still_screen::run g;
    for (std::uint64_t tick = 0; tick <= 1000; tick += 100) g.observe(sample(tick, 0.f));
    o = g.observe(sample(1000 + still::max_gap_ms, 0.f));
    require(!o.short_run && g.run_ms() == 1000 + still::max_gap_ms, "A gap of max_gap_ms restarted the run");
    o = g.observe(sample(1000 + 2 * still::max_gap_ms + 1, 0.f));
    require(o.short_run && o.short_ms == 1000 + still::max_gap_ms && g.run_ms() == 0, "A gap past max_gap_ms kept the run");
    for (std::uint64_t tick = 2100; tick < 4000; tick += 100) require(!g.observe(sample(tick, 0.f)).entered, "Entered early");
    require(g.observe(sample(4001, 0.f)).entered, "The restarted run did not enter after its own 2 s");
    // Active: a stall does not release it, the next failing sample does.
    o = g.observe(sample(6000, 0.f));
    require(!o.ended && g.active(), "A gap released an active run");
    require(g.observe(sample(6100, .3f)).ended, "A failing sample after a gap did not release");
    std::puts("PASS H2 density: a pending run restarts after a gap over 500 ms; two samples 2.5 s apart never enter");
  }

  // Decision words of one completed sample: texel 5 {n, D, state}, texel 9's
  // reused bit, texel 12 {still, compared}.
  std::array<std::uint32_t, 4 * ui_detection::still_decision_texels> words(std::uint32_t source, std::uint32_t n, float d,
      bool depth_current, std::uint32_t still_cells, std::uint32_t compared, bool reused = false) {
    namespace word = ui_detection::decision_word;
    std::array<std::uint32_t, 4 * ui_detection::still_decision_texels> t{};
    t[word::source] = source;
    t[word::pixels] = 3840u * 2160u;
    t[word::candidates] = ui_detection::candidate::layer | ui_detection::candidate::current;
    t[word::scene_n] = n;
    std::memcpy(&t[word::scene_d], &d, sizeof(d));
    // Reused depth: the evidence ran but is not valid (no H1 verdict).
    const auto verdict = depth_current ? ui_detection::scene_verdict_of(d) : ui_detection::scene_verdict::none;
    t[word::scene_state] = (depth_current ? 1u : 0u) | 2u | std::uint32_t(verdict) << 2;
    t[word::scene_decided] = 4u * n;
    t[word::frame_reason] = ui_detection::frame_reason_decided | (reused ? ui_detection::frame_reason_reused : 0u);
    t[word::still_cells] = still_cells;
    t[word::still_compared] = compared;
    return t;
  }

  void guard_decodes_texel_12() {
    const auto t = words(0, 405, -.014f, false, share(99), cells, true);
    const auto s = scene_guard::sample_of(t.data(), t.size(), 1234, true);
    require(s.still_cells == share(99) && s.still_compared == cells && s.reused && s.presented.ran && !s.presented.valid &&
        s.presented.n == 405 && s.presented.d == -.014f, "The guard did not decode texel 12, the reused bit or invalid evidence");
    const auto older = scene_guard::sample_of(t.data(), 4 * ui_detection::pre_ui_decision_texels, 1234, true);
    require(!older.still_cells && !older.still_compared, "The guard decoded texel 12 from fewer texels");
    // Measurement: H2 runs the evidence passes without making them
    // actionable.
    scene_guard::state guard;
    guard.enter_scope(1, 1);
    const auto idle = guard.measure(1000, false, false, false), measured = guard.measure(1000, false, false, false, true);
    require(!idle.run && measured.run && !measured.actionable, "H2's measurement was not a non-actionable run");
    std::puts("PASS H2 guard decode: texel 12 and the reused bit; reused-depth evidence keeps n and D; measure(still) runs");
  }

  // decide() with the flag the renderer pushes while the run is active.
  void loading_frame_shows_flat() {
    ui_selection::counts c;
    c.pixels = 1000;
    c.covered = {0, 0, 0, 0, 0};
    c.invalid = {0, 0, 600, 0, 0};
    const auto d = ui_selection::decide(c, 0x48, 0x00, 0, {}, still::flatten);
    require(d.source == ui_detection::source_still && d.covered == 1000 && d.still && d.own_source == 0 &&
        d.next.state == ui_detection::hold::own && d.next.source == 0, "decide() did not show the loading frame flat as 11");
  }

  // Stellar Blade SDR loading, sb_s2b_0719.log 07:19:57.1-07:20:06.0: the
  // offered layer 0x48 is a bare cleared target and the current alpha is
  // unaccepted, so no source decides; D read -0.014 to -0.068 at n 405-491,
  // and about 12% of the samples consumed reused depth (texel 5 not valid).
  // The screen is assumed 100% still (not measurable offline).
  void stellar_blade_loading() {
    scene_guard::state guard;
    guard.enter_scope(1, 1);
    const auto sigs = scene_guard::kind_signatures{};
    const std::array<float, 10> d_by_second{-.014f, -.020f, -.030f, -.045f, -.068f, -.044f, -.047f, -.049f, -.049f, -.040f};
    std::uint64_t shadow_entered = 0;
    unsigned reused_depth = 0, samples = 0;
    // Shadow: the run enters, the decision stays (nothing pushed).
    for (std::uint64_t tick = 57100; tick <= 66000; tick += 100, ++samples) {
      const bool depth_current = samples % 8 != 3; // 12.5%: reused depth.
      reused_depth += !depth_current;
      const float d = d_by_second[std::min<std::size_t>((tick - 57000) / 1000, d_by_second.size() - 1)];
      const auto t = words(0, 405 + std::uint32_t(tick % 87), d, depth_current, cells, samples ? cells : 0u);
      const auto observed = guard.observe(scene_guard::sample_of(t.data(), t.size(), tick, true), false, sigs, true);
      if (observed.still.entered) shadow_entered = tick;
      require(!observed.still.ended && !observed.still.short_run, "The loading screen's run ended");
    }
    // The first sample has no previous cell means (unmeasured), so the run
    // starts at 57200 and enters at 59200.
    require(shadow_entered == 59200 && guard.still_flatten() && reused_depth * 100 >= samples * 12,
      "The loading screen did not enter at 2 s, or reused depth reset it");
    // Enabled: the renderer pushes still::flatten from the first render
    // after the entering sample is read; decide shows that frame flat (11).
    loading_frame_shows_flat();
    // Loading ends: the first moving sample releases it.
    const auto t = words(0, 600, .02f, true, share(80), cells);
    const auto observed = guard.observe(scene_guard::sample_of(t.data(), t.size(), 66100, true), false, sigs, true);
    require(observed.still.ended && observed.still.reason == end_reason::moving && !guard.still_flatten() &&
        observed.still.ended_episode.duration_ms() == 66000 - 57200,
      "The first moving sample after loading did not release the run");
    std::printf("PASS H2 Stellar Blade SDR loading: %u samples (%u on reused depth) enter at 2.0 s and hold to the end; "
      "the first moving sample releases\n", samples, reused_depth);
  }
  // Expedition 33 SDR FG-off gameplay (current alpha unaccepted): 1 s
  // samples read D 0.380, 0.025, 0.424, 0.035, 0.211, 0.112, 0.193, 0.074,
  // 0.539, 0.099, 0.229, 0.072, 0.166, 0.040, 0.362, 0.111, 0.320; the
  // first-run shadow's hidden runs lasted at most 547 ms. Even assuming the
  // frame 100% still, isolated low runs never reach 2 s.
  void expedition_33_dips() {
    const std::array<float, 17> d_by_second{.380f, .025f, .424f, .035f, .211f, .112f, .193f, .074f, .539f, .099f, .229f, .072f,
      .166f, .040f, .362f, .111f, .320f};
    still_screen::run r;
    std::uint64_t longest = 0;
    unsigned shorts = 0;
    for (std::uint64_t tick = 2000; tick < 19000; tick += 100) {
      const auto second = (tick - 2000) / 1000, within = tick % 1000;
      // A low second holds its D for at most 500 ms around the 1 s sample
      // (the measured hidden runs reached 547 ms at D < 0.15).
      const float d = d_by_second[second] <= .05f && within >= 300 && within <= 800 ? d_by_second[second] : .3f;
      const auto o = r.observe(sample(tick, d));
      require(!o.entered && !r.active(), "An E33 gameplay dip entered H2");
      if (o.short_run) {
        ++shorts;
        longest = std::max(longest, o.short_ms);
      }
    }
    require(shorts == 3 && longest == 500, "The E33 dips were not counted as three short runs of 500 ms");
    std::printf("PASS H2 Expedition 33 SDR gameplay dips: %u short runs, longest %llu ms, never entered\n", shorts,
      static_cast<unsigned long long>(longest));
  }
}

int main() {
  try {
    scope_is_sdr_output();
    predicate_edges();
    run_enters_and_releases();
    leave_and_identity();
    run_needs_dense_samples();
    guard_decodes_texel_12();
    stellar_blade_loading();
    expedition_33_dips();
    std::puts("Still screens (H2): 8 groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
