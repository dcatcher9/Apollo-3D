// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_alpha_auto.h"
#include "game3d_scene_guard.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_selection.h"
#include "game3d_ui_temporal.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace sunshine_game3d;
  using kind = ui_selection::kind;
  namespace candidate = ui_detection::candidate;
  constexpr std::uint32_t pixels = 1000;

  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }

  // Each kind's typed format: UIAlpha R8 (61), the UI color tag BGRA8 (87),
  // the layer R16G16B16A16_FLOAT (10), Backbuffer, current and HUD-less
  // R10G10B10A2 (24); color space 1 sRGB, 3 PQ.
  candidate_signatures signatures(std::uint32_t color_space = 1) {
    candidate_signatures result;
    result.color_space = color_space;
    result.set(kind::ui_alpha, 61).set(kind::ui_color, 87).set(kind::ui_layer, 10).set(kind::backbuffer, 24)
      .set(kind::current, 24).set(kind::hudless, 24);
    return result;
  }
  bool accepts(alpha_auto_policy &policy, kind k, std::uint32_t color_space = 1) {
    return policy.accepts(signatures(color_space).of(k));
  }

  // One detection sample's evidence over `pixels` pixels.
  struct sample {
    alpha_auto_decision::detection_evidence e;
    explicit sample(std::uint32_t offered = 0) { e.candidates = offered; }
    sample &alpha(kind k, std::uint32_t covered, std::uint32_t invalid = 0) {
      e.candidates |= ui_selection::bit(k);
      switch (k) {
        case kind::ui_alpha: e.alpha_covered[0] = covered; e.alpha_invalid[0] = invalid; break;
        case kind::ui_color: e.alpha_covered[1] = covered; e.alpha_invalid[1] = invalid; break;
        case kind::backbuffer: e.alpha_covered[2] = covered; e.alpha_invalid[2] = invalid; break;
        case kind::current: e.alpha_covered[3] = covered; e.alpha_invalid[3] = invalid; break;
        case kind::ui_layer: e.layer_covered = covered; e.layer_invalid = invalid; break;
        default: break;
      }
      return *this;
    }
    // A HUD-less pair: changed and unchanged pixels, exact or not.
    sample &pair(std::uint32_t changed, std::uint32_t unchanged, bool exact = true, std::uint32_t tiles = 256) {
      e.candidates |= candidate::hudless | (exact ? candidate::exact : 0u);
      e.hudless_changed = changed;
      e.hudless_unchanged = unchanged;
      e.matching_tiles = tiles;
      e.hudless_lit = pixels;
      return *this;
    }
    // A2's one-way counts of a judged kind (layer, Backbuffer, current):
    // pixels with alpha of at least 1/2, and those of them where the HUD-less
    // image is lit and unchanged.
    sample &one_way(kind k, std::uint32_t strong, std::uint32_t contradicted) {
      for (std::size_t i = 0; i != ui_selection::judged_kinds.size(); ++i)
        if (ui_selection::judged_kinds[i] == k) {
          e.strong[i] = strong;
          e.contradicted[i] = contradicted;
        }
      return *this;
    }
    // The offered layer was the one-frame-late copy (stored flag 0x4), for
    // which the GPU counts no strong pixel.
    sample &late() {
      e.late_layer = true;
      return *this;
    }
  };
  // The offered candidates that pass V1/V2, as the GPU reports them in the
  // sample (ui_selection::decide).
  std::uint32_t valid_bits_of(const alpha_auto_decision::detection_evidence &e) {
    ui_selection::counts c;
    c.pixels = pixels;
    c.covered = {e.alpha_covered[0], e.alpha_covered[1], e.layer_covered, e.alpha_covered[2], e.alpha_covered[3]};
    c.invalid = {e.alpha_invalid[0], e.alpha_invalid[1], e.layer_invalid, e.alpha_invalid[2], e.alpha_invalid[3]};
    c.changed = e.hudless_changed;
    c.unchanged = e.hudless_unchanged;
    c.nonfinite = e.hudless_invalid;
    c.lit = e.hudless_lit;
    c.matching_tiles = e.matching_tiles;
    return ui_selection::decide(c, e.candidates, 0u, 0u).valid_bits;
  }
  void feed(alpha_auto_policy &policy, const sample &s, std::uint64_t tick, std::uint32_t color_space = 1) {
    auto evidence = s.e;
    evidence.valid_bits = valid_bits_of(evidence);
    policy.observe(evidence, pixels, tick, signatures(color_space));
  }

  void mode_is_auto_until_a_manual_edit_and_auto_again_after_it() {
    alpha_auto_policy policy;
    const auto fresh = policy.decision();
    require(!fresh.enabled && fresh.state == alpha_auto_state::waiting_for_source, "A new session did not start in Auto");
    policy.set_manual(true);
    const auto on = policy.decision();
    require(on.enabled && on.state == alpha_auto_state::manual_on, "Manual On was not reported");
    policy.set_manual(false);
    const auto off = policy.decision();
    require(!off.enabled && off.state == alpha_auto_state::manual_off, "Manual Off was not reported");
    policy.set_automatic();
    const auto automatic = policy.decision();
    require(!automatic.enabled && automatic.state == alpha_auto_state::waiting_for_source, "Returning to Auto kept the manual mode");
  }

  // A1: a declared source (UIAlpha, the UI color tag) is accepted by its first
  // valid selective sample; a full, empty, invalid or void sample is none.
  void declared_sources_are_accepted_by_one_selective_sample() {
    alpha_auto_policy policy;
    feed(policy, sample().alpha(kind::ui_color, 1000), 1000);        // Full (an opaque final image).
    feed(policy, sample().alpha(kind::ui_color, 0), 1100);           // Empty.
    feed(policy, sample().alpha(kind::ui_color, 26, 11), 1200);      // More than 1% invalid.
    feed(policy, sample().alpha(kind::ui_color, 26).alpha(kind::ui_alpha, 30, 20), 1300); // Void: UIAlpha invalid.
    require(!accepts(policy, kind::ui_color) && !policy.accepted(0x3fu, signatures()),
      "A full, empty, invalid or void sample accepted a declared source");
    feed(policy, sample().alpha(kind::ui_color, 26, 10), 1400);      // 1% invalid is valid.
    require(accepts(policy, kind::ui_color) && policy.counters()[ui_counter::trust_earned] == 1,
      "One valid selective sample did not accept the UI color tag");
    require(policy.accepted(candidate::ui_color | candidate::backbuffer, signatures()) == candidate::ui_color &&
        !policy.accepted(candidate::ui_alpha, signatures()), "Acceptance leaked to another candidate");
    require(policy.stored() == "ui_color:87:srgb", "The accepted key was not stored");
    // A never-selective source is never accepted (Stellar Blade's opaque tag
    // in HDR with FG on): full on every sample for a minute.
    alpha_auto_policy opaque;
    for (std::uint64_t tick = 1000; tick <= 60000; tick += 100)
      feed(opaque, sample().alpha(kind::ui_color, 1000).alpha(kind::ui_layer, 18).pair(900, 100), tick, 3);
    require(!accepts(opaque, kind::ui_color, 3) && accepts(opaque, kind::ui_layer, 3),
      "A never-selective tag was accepted, or the layer beside it was not");
  }

  // A1: an inferred source (layer, Backbuffer, current alpha) needs
  // alpha_trust_samples valid selective samples spanning alpha_trust_span_ms
  // within a factor of two; up to 1% invalid pixels count.
  void inferred_sources_need_a_steady_run() {
    alpha_auto_policy policy;
    feed(policy, sample().alpha(kind::backbuffer, 200), 10000);
    feed(policy, sample().alpha(kind::backbuffer, 200, 10), 11000);
    feed(policy, sample().alpha(kind::backbuffer, 200), 11999);
    require(!accepts(policy, kind::backbuffer), "Three samples within 2 s accepted an inferred source");
    feed(policy, sample().alpha(kind::backbuffer, 200), 12000);
    require(accepts(policy, kind::backbuffer) && !accepts(policy, kind::current),
      "Selective samples over 2 s did not accept exactly their source");
    // Mode edits do not forget acceptance.
    policy.set_manual(false);
    policy.set_automatic();
    require(accepts(policy, kind::backbuffer), "Mode edits forgot what the game session accepted");
    // More than 1% invalid is no evidence.
    alpha_auto_policy noisy;
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 1000) feed(noisy, sample().alpha(kind::current, 200, 11), tick);
    require(!accepts(noisy, kind::current), "Invalid presented alpha was accepted");
    // A backwards clock restarts the interval.
    alpha_auto_policy clock;
    feed(clock, sample().alpha(kind::ui_layer, 100), 5000);
    feed(clock, sample().alpha(kind::ui_layer, 100), 4000);
    feed(clock, sample().alpha(kind::ui_layer, 100), 5500);
    feed(clock, sample().alpha(kind::ui_layer, 100), 5999);
    require(!accepts(clock, kind::ui_layer), "A backwards clock kept the earlier interval");
    feed(clock, sample().alpha(kind::ui_layer, 100), 6000);
    require(accepts(clock, kind::ui_layer), "The restarted interval did not accept the layer");
    // A full sample restarts the run; an empty one neither earns nor restarts.
    alpha_auto_policy run;
    feed(run, sample().alpha(kind::current, 200), 1000);
    feed(run, sample().alpha(kind::current, 200), 2000);
    feed(run, sample().alpha(kind::current, 1000), 2500);
    feed(run, sample().alpha(kind::current, 200), 3000);
    require(!accepts(run, kind::current), "A full-frame sample did not restart the earning run");
    feed(run, sample().alpha(kind::current, 0), 3500);
    feed(run, sample().alpha(kind::current, 300), 4000);
    feed(run, sample().alpha(kind::current, 250), 5000);
    require(accepts(run, kind::current), "Consecutive steady samples did not accept current alpha");
    // Coverage that swings beyond a factor of two (scene effects) never earns.
    alpha_auto_policy effects;
    for (std::uint64_t tick = 1000; tick <= 20000; tick += 500)
      feed(effects, sample().alpha(kind::ui_layer, tick % 1000 ? 50u : 400u), tick);
    require(!accepts(effects, kind::ui_layer), "Fluctuating coverage was accepted");
  }

  // A1 void: while an offered declared alpha is V1-invalid, no source's
  // earning advances or restarts (Resident Evil Requiem's rejected-tag
  // frames, where presented alpha covered 64-80%).
  void an_invalid_declared_alpha_voids_earning() {
    alpha_auto_policy policy;
    feed(policy, sample().alpha(kind::backbuffer, 200), 1000);
    feed(policy, sample().alpha(kind::backbuffer, 200), 2000);
    for (std::uint64_t tick = 2100; tick <= 9000; tick += 100)
      feed(policy, sample().alpha(kind::backbuffer, tick % 200 ? 200u : 1000u).alpha(kind::ui_color, 0, 600), tick);
    require(!accepts(policy, kind::backbuffer) && !accepts(policy, kind::ui_color),
      "A sample beside an invalid tag earned acceptance");
    // The void neither advanced nor restarted the run: the third clean sample
    // completes it.
    feed(policy, sample().alpha(kind::backbuffer, 200), 9100);
    require(accepts(policy, kind::backbuffer), "A void sample restarted the earning run");
    // Revocation still evaluates void samples: a valid exact pair one-way
    // contradicts the full Backbuffer claim.
    for (const std::uint64_t tick : {10000u, 11000u, 12000u})
      feed(policy, sample().alpha(kind::backbuffer, 1000).alpha(kind::ui_color, 0, 600).pair(10, 900)
        .one_way(kind::backbuffer, 1000, 800), tick);
    require(!accepts(policy, kind::backbuffer) && policy.counters()[ui_counter::trust_revoked_exact] == 1,
      "A one-way contradiction in void samples was not revoked");
  }

  // A1 key: kind, typed format and the swapchain color space; FG mode is not
  // part of it.
  void acceptance_is_per_signature() {
    alpha_auto_policy policy;
    for (const std::uint64_t tick : {1000u, 2000u, 3000u}) feed(policy, sample().alpha(kind::backbuffer, 26), tick, 3);
    require(accepts(policy, kind::backbuffer, 3) && !accepts(policy, kind::backbuffer, 1),
      "Acceptance crossed the color space");
    require(policy.accepted(candidate::backbuffer, signatures(3)) == candidate::backbuffer &&
        !policy.accepted(candidate::backbuffer, signatures(1)), "accepted() crossed the color space");
    auto other_format = signatures(3);
    other_format.set(kind::backbuffer, 87);
    require(!policy.accepted(candidate::backbuffer, other_format), "accepted() crossed the format");
    // In sRGB it earns again on its own; the PQ entry is unchanged.
    for (const std::uint64_t tick : {4000u, 5000u, 6000u}) feed(policy, sample().alpha(kind::backbuffer, 26), tick, 1);
    require(accepts(policy, kind::backbuffer, 1) && accepts(policy, kind::backbuffer, 3),
      "The same kind and format in sRGB and PQ were not independent");
    require(policy.stored() == "backbuffer:24:srgb,backbuffer:24:pq", "Keys were not stored in signature order");
    // Keys round-trip; anything else is none.
    for (const auto &s : {ui_selection::signature{kind::ui_layer, 10, 3}, ui_selection::signature{kind::hudless, 87, 1},
             ui_selection::signature{kind::current, 24, 4}}) {
      const auto parsed = ui_selection::signature::parse(s.key());
      require(parsed && *parsed == s, "A signature key did not round-trip");
    }
    for (const char *legacy : {"4", "16", "0x1f", "", "ui_color:87", "ui_color:x:srgb", "frame:87:srgb", "ui_color:87:srgb:fg"})
      require(!ui_selection::signature::parse(legacy), "A legacy or malformed entry parsed as a signature");
  }

  // A3 and the legacy discard: restored entries decide at once but lapse
  // unless earned again within alpha_trust_reconfirm_ms of first being
  // offered valid, a clock that pauses while a declared source is offered but
  // invalid; anything that is not a key is discarded and counted.
  void restore_is_provisional_and_legacy_entries_are_discarded() {
    alpha_auto_policy legacy;
    for (const char *stored : {"4", "16", "0x1f"}) {
      alpha_auto_policy one;
      const auto result = one.restore(stored);
      require(!result.restored && result.discarded == 1 && result.discarded_text == stored &&
          one.counters()[ui_counter::trust_discarded] == 1 && one.stored().empty(), "A legacy entry was not discarded");
    }
    // ReShade returns an array's elements separated by '\0', with the end
    // padded; a mix of legacy and new entries keeps the new ones, once.
    const char mixed[] = "4\0ui_color:87:srgb,backbuffer:24:pq\0 ui_color:87:srgb \0\0";
    std::vector<std::string> heard;
    legacy.on_change([&](const std::string &stored) { heard.push_back(stored); });
    const auto result = legacy.restore(std::string_view(mixed, sizeof(mixed)));
    require(result.restored == 2 && result.discarded == 1 && result.discarded_text == "4" &&
        legacy.counters()[ui_counter::trust_restored] == 2 && legacy.counters()[ui_counter::trust_discarded] == 1,
      "A mix of legacy and new entries was not restored and discarded");
    require(legacy.stored() == "ui_color:87:srgb,backbuffer:24:pq" && heard.empty(),
      "Restored keys were not kept, or the restore was reported");
    require(legacy.accepted(candidate::ui_color | candidate::backbuffer, signatures(1)) == candidate::ui_color &&
        legacy.accepted(candidate::backbuffer, signatures(3)) == candidate::backbuffer, "Restored keys did not decide at once");
    alpha_auto_policy capped;
    std::string many;
    for (std::uint32_t format = 1; format <= 40; ++format) many += "current:" + std::to_string(format) + ":srgb,";
    require(capped.restore(many).restored == max_stored_ui_sources, "More than the cap was restored");

    // Lapse: a minute after first being offered valid, unless earned again.
    alpha_auto_policy remembered;
    remembered.restore("current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 60000; tick += 1000)
      feed(remembered, sample().alpha(kind::current, tick % 2000 ? 500u : 50u), tick);
    require(accepts(remembered, kind::current), "Restored acceptance lapsed before a minute");
    feed(remembered, sample().alpha(kind::current, 50), 61000);
    require(!accepts(remembered, kind::current) && remembered.counters()[ui_counter::trust_lapsed] == 1,
      "Unconfirmed restored acceptance did not lapse");
    // The clock starts when the source is first offered valid, not at restore.
    alpha_auto_policy late;
    late.restore("current:24:srgb");
    feed(late, sample().alpha(kind::backbuffer, 0), 1000);
    feed(late, sample().alpha(kind::current, 500, 600), 2000);
    feed(late, sample().alpha(kind::current, 0), 30000);
    feed(late, sample().alpha(kind::current, 0), 89999);
    require(accepts(late, kind::current), "The lapse clock started before the source was offered valid");
    feed(late, sample().alpha(kind::current, 0), 90000);
    require(!accepts(late, kind::current), "The lapse clock did not start at the first valid offer");
    // A sample opaque almost everywhere (a full menu: Stellar Blade's presented
    // alpha while frame generation is suspended) can neither earn nor refute
    // the source, so the clock pauses there; empty samples run it again.
    alpha_auto_policy menus;
    menus.restore("current:24:srgb");
    feed(menus, sample().alpha(kind::current, 0), 1000);
    for (std::uint64_t tick = 2000; tick <= 200000; tick += 1000) feed(menus, sample().alpha(kind::current, 1000), tick);
    require(accepts(menus, kind::current) && menus.counters()[ui_counter::trust_lapsed] == 0,
      "Full menus ran the reconfirm clock of a restored source");
    feed(menus, sample().alpha(kind::current, 0), 201000);
    feed(menus, sample().alpha(kind::current, 0), 259999);
    require(accepts(menus, kind::current), "The clock counted the paused full menus");
    feed(menus, sample().alpha(kind::current, 0), 260000);
    require(!accepts(menus, kind::current), "An unconfirmed source did not lapse after 60 s of testable time");
    // A restored declared source confirmed by one selective sample keeps it.
    alpha_auto_policy confirmed;
    confirmed.restore("ui_alpha:61:srgb");
    feed(confirmed, sample().alpha(kind::ui_alpha, 100), 1000);
    for (std::uint64_t tick = 2000; tick <= 200000; tick += 1000) feed(confirmed, sample().alpha(kind::ui_alpha, 0), tick);
    require(accepts(confirmed, kind::ui_alpha) && confirmed.counters()[ui_counter::trust_earned] == 1,
      "Acceptance earned again in this session lapsed later, or was not counted");

    // A restored declared tag offered but invalid (Resident Evil Requiem's
    // rejected-tag frames) for longer than the lapse pauses its clock, from
    // the first invalid offer to the next valid one: it lapses once its valid
    // offers have run the clock for a minute in total.
    alpha_auto_policy rejected;
    rejected.restore("ui_color:87:srgb");
    feed(rejected, sample().alpha(kind::ui_color, 0), 1000);
    for (std::uint64_t tick = 2000; tick <= 100000; tick += 1000)
      feed(rejected, sample().alpha(kind::ui_color, 0, 600).alpha(kind::current, 400), tick);
    require(accepts(rejected, kind::ui_color) && !rejected.counters()[ui_counter::trust_lapsed],
      "A restored tag lapsed while it was offered but invalid");
    feed(rejected, sample().alpha(kind::ui_color, 0), 100500);
    feed(rejected, sample().alpha(kind::ui_color, 0), 159499);
    require(accepts(rejected, kind::ui_color), "The paused clock did not resume at the next valid offer");
    feed(rejected, sample().alpha(kind::ui_color, 0), 159500);
    require(!accepts(rejected, kind::ui_color) && rejected.counters()[ui_counter::trust_lapsed] == 1,
      "The paused clock never lapsed");
    // Invalid offers never extend the clock: a tag invalid every other
    // second lapses after 60 s of valid offers in total.
    alpha_auto_policy alternating;
    alternating.restore("ui_color:87:srgb");
    for (std::uint64_t tick = 1000; tick <= 120000; tick += 1000)
      feed(alternating, (tick / 1000) % 2 ? sample().alpha(kind::ui_color, 0) : sample().alpha(kind::ui_color, 0, 600), tick);
    require(accepts(alternating, kind::ui_color), "An alternating tag lapsed before 60 s of valid offers");
    feed(alternating, sample().alpha(kind::ui_color, 0), 121000);
    require(!accepts(alternating, kind::ui_color) && alternating.counters()[ui_counter::trust_lapsed] == 1,
      "Invalid offers kept restarting the clock of a tag that never confirmed");
    alpha_auto_policy recovered;
    recovered.restore("ui_color:87:srgb");
    feed(recovered, sample().alpha(kind::ui_color, 1000), 1000);
    for (std::uint64_t tick = 2000; tick <= 100000; tick += 1000) feed(recovered, sample().alpha(kind::ui_color, 0, 600), tick);
    feed(recovered, sample().alpha(kind::ui_color, 30), 100500);
    for (std::uint64_t tick = 101000; tick <= 300000; tick += 1000) feed(recovered, sample().alpha(kind::ui_color, 1000), tick);
    require(accepts(recovered, kind::ui_color) && recovered.counters()[ui_counter::trust_earned] == 1,
      "A tag that confirmed after its invalid run was not kept");
    // A restored HUD-less pair offered without its valid bit (a middle-band
    // change set) is invalid the same way.
    alpha_auto_policy pair;
    pair.restore("hudless:24:srgb");
    feed(pair, sample().pair(980, 10), 1000);
    for (std::uint64_t tick = 2000; tick <= 100000; tick += 1000) feed(pair, sample().pair(500, 500), tick);
    require(accepts(pair, kind::hudless), "A restored HUD-less pair lapsed while it was invalid");
    feed(pair, sample().pair(980, 10), 100500);
    feed(pair, sample().pair(980, 10), 160500);
    require(!accepts(pair, kind::hudless), "A restored HUD-less pair never lapsed after its invalid run");
    // Inferred entries keep the clock: invalid samples do not re-arm it.
    alpha_auto_policy inferred;
    inferred.restore("current:24:srgb");
    feed(inferred, sample().alpha(kind::current, 0), 1000);
    for (std::uint64_t tick = 2000; tick <= 60000; tick += 1000) feed(inferred, sample().alpha(kind::current, 0, 600), tick);
    feed(inferred, sample().alpha(kind::current, 0), 61000);
    require(!accepts(inferred, kind::current), "Invalid samples re-armed an inferred entry's clock");
  }

  // Forget (A3): clears every entry of the game, accepted, provisional,
  // earning and in doubt; the listener hears "" and trust.forgotten counts
  // the accepted signatures. Each source then earns again by its own rule.
  void forget_clears_the_ledger() {
    alpha_auto_policy policy;
    std::vector<std::string> heard;
    policy.on_change([&](const std::string &stored) { heard.push_back(stored); });
    require(policy.forget().empty() && heard.empty() && !policy.counters()[ui_counter::trust_forgotten],
      "Forgetting nothing was reported");
    policy.restore("ui_layer:10:srgb");
    feed(policy, sample().alpha(kind::ui_color, 30), 1000);
    feed(policy, sample().alpha(kind::current, 200), 1000);
    feed(policy, sample().alpha(kind::current, 200), 2000);
    require(policy.stored() == "ui_color:87:srgb,ui_layer:10:srgb", "The ledger did not hold the earned and restored keys");
    heard.clear();
    require(policy.forget() == "ui_color:87:srgb,ui_layer:10:srgb" && heard == std::vector<std::string>{""} &&
        policy.counters()[ui_counter::trust_forgotten] == 2 && policy.stored().empty() &&
        !policy.accepted(0x5fu, signatures()), "Forget did not clear, report or count the accepted keys");
    // The earning run was cleared too: one more sample does not complete it.
    feed(policy, sample().alpha(kind::current, 200), 3000);
    require(!accepts(policy, kind::current), "Forget kept an earning run");
    // Each source earns again by its own rule.
    feed(policy, sample().alpha(kind::ui_color, 30), 4000);
    require(accepts(policy, kind::ui_color) && heard.back() == "ui_color:87:srgb", "A forgotten tag could not earn again");
    feed(policy, sample().alpha(kind::current, 200), 4000);
    feed(policy, sample().alpha(kind::current, 200), 5000);
    require(accepts(policy, kind::current), "A forgotten inferred source could not earn again by its run");
    // Forget leaves the mode and the first-run shadow alone.
    policy.set_first_run(true);
    policy.set_manual(true);
    policy.forget();
    require(policy.first_run() && policy.decision().state == alpha_auto_state::manual_on && policy.stored().empty(),
      "Forget changed the mode or the first-run shadow");
  }

  // A2: an accepted source is revoked only by same-sample evidence of
  // stronger provenance, whatever the drawing rank: a V2-valid exact change
  // set's one-way test, or the coverage of an accepted, valid declared
  // alpha, over the A1 evidence interval. Per signature.
  void revocation_is_per_signature() {
    alpha_auto_policy policy;
    std::vector<std::string> heard;
    policy.on_change([&](const std::string &stored) { heard.push_back(stored); });
    for (const std::uint64_t tick : {1000u, 2000u, 3000u}) feed(policy, sample().alpha(kind::backbuffer, 200), tick);
    feed(policy, sample().alpha(kind::backbuffer, 200), 3000, 3);
    require(accepts(policy, kind::backbuffer) && heard == std::vector<std::string>{"backbuffer:24:srgb"},
      "Earning was not reported exactly once");
    // No contradiction: a full claim over a middle-band (V2-invalid) pair, an
    // inexact pair or no pair is unjudged; over a valid exact pair whose lit
    // unchanged pixels it does not cover (a dim or tint over dark or changed
    // pixels: the Expedition 33 pause, The Witcher 3 sign wheel) or covers
    // with less than a tenth of its strong pixels, it is judged and agrees.
    for (std::uint64_t tick = 4000; tick <= 9000; tick += 1000) {
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(500, 500).one_way(kind::backbuffer, 1000, 1000), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900, false).one_way(kind::backbuffer, 1000, 900), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).one_way(kind::backbuffer, 1000, 0), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 1000, 0), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 1000, 99), tick);
    }
    require(accepts(policy, kind::backbuffer) && !policy.counters()[ui_counter::trust_revoked_exact],
      "A middle-band or inexact pair, or a dim over unlit or changed pixels, revoked acceptance");
    // Three contradictions within 2 s revoke; older ones drop out, and
    // agreeing or unjudged samples between them change nothing, nor does a
    // judged source without strong pixels. (The exact selective pairs earn
    // the HUD-less pair meanwhile.)
    const auto contradicted = sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 1000, 800);
    feed(policy, contradicted, 10000);
    feed(policy, contradicted, 12001);
    require(accepts(policy, kind::backbuffer), "Contradictions more than 2 s apart revoked acceptance");
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 1000, 0), 12500);
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 0, 0), 12600);
    feed(policy, sample().alpha(kind::backbuffer, 1000), 13000);
    // The same contradiction in another color space doubts only that signature.
    feed(policy, contradicted, 13500, 3);
    feed(policy, contradicted, 13999, 3);
    feed(policy, contradicted, 14000);
    require(accepts(policy, kind::backbuffer), "Two contradictions within 2 s revoked acceptance");
    feed(policy, contradicted, 14001);
    require(!accepts(policy, kind::backbuffer) && policy.counters()[ui_counter::trust_revoked_exact] == 1 &&
        !policy.counters()[ui_counter::trust_revoked_declared] && heard.back() == "hudless:24:srgb,hudless:24:pq",
      "Three one-way contradictions within 2 s were not revoked, counted and reported");
    // A contradicted sample earns nothing and restarts the earning run.
    const auto selective_contradicted = sample().alpha(kind::backbuffer, 200).pair(100, 900).one_way(kind::backbuffer, 200, 150);
    for (std::uint64_t tick = 15000; tick <= 20000; tick += 500) {
      feed(policy, sample().alpha(kind::backbuffer, 200), tick);
      feed(policy, selective_contradicted, tick + 250);
    }
    require(!accepts(policy, kind::backbuffer), "A contradicted source earned acceptance again");
    for (const std::uint64_t tick : {21000u, 22000u, 23000u}) feed(policy, sample().alpha(kind::backbuffer, 200), tick);
    require(accepts(policy, kind::backbuffer), "An uncontradicted run did not earn acceptance again");
    // The one-frame-late layer copy is not same-sample evidence (E2): the GPU
    // counts no strong pixel of it, and neither judge reads it, not even a
    // disagreeing accepted UIAlpha.
    alpha_auto_policy late;
    late.restore("ui_layer:10:srgb,ui_alpha:61:srgb");
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 500)
      feed(late, sample().alpha(kind::ui_layer, 1000).alpha(kind::ui_alpha, 20).pair(100, 900).late(), tick);
    require(accepts(late, kind::ui_layer) && !late.counters()[ui_counter::trust_revoked_exact] &&
        !late.counters()[ui_counter::trust_revoked_declared], "The late layer was judged");

    // Presented alpha disagreeing with an accepted UIAlpha (Resident Evil
    // Requiem: tag 0.2%, presented alpha 35-100% in play; menus agree).
    alpha_auto_policy presented;
    presented.restore("ui_alpha:61:srgb,current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 4000; tick += 1000)
      feed(presented, sample().alpha(kind::ui_alpha, 1000).alpha(kind::current, 950), tick);
    require(accepts(presented, kind::current), "Presented alpha agreeing with a full-screen menu was contradicted");
    feed(presented, sample().alpha(kind::ui_alpha, 2).alpha(kind::current, 999), 5000);
    feed(presented, sample().alpha(kind::ui_alpha, 0).alpha(kind::current, 400), 6000);
    require(accepts(presented, kind::current), "Two samples revoked acceptance");
    feed(presented, sample().alpha(kind::ui_alpha, 2).alpha(kind::current, 350), 7000);
    require(!accepts(presented, kind::current) && accepts(presented, kind::ui_alpha) &&
        presented.counters()[ui_counter::trust_revoked_declared] == 1 && !presented.counters()[ui_counter::trust_revoked_exact],
      "A declared UI mask did not revoke presented alpha that covers the scene");
    for (std::uint64_t tick = 8000; tick <= 13500; tick += 500)
      feed(presented, sample().alpha(kind::ui_alpha, tick < 10500 ? 2u : 1000u).alpha(kind::current, tick < 10500 ? 300u : 4u), tick);
    require(!accepts(presented, kind::current), "Contradicted presented alpha earned acceptance again");
    for (const std::uint64_t tick : {14000u, 15000u, 16000u})
      feed(presented, sample().alpha(kind::ui_alpha, 20).alpha(kind::current, 30), tick);
    require(accepts(presented, kind::current), "Presented alpha agreeing with the UI mask could not earn acceptance");
    // The layer is inferred: an accepted UIAlpha judges it by coverage, and
    // it judges nothing itself.
    alpha_auto_policy layer;
    layer.restore("ui_alpha:61:srgb,ui_layer:10:srgb");
    for (const std::uint64_t tick : {1000u, 2000u, 3000u})
      feed(layer, sample().alpha(kind::ui_alpha, 20).alpha(kind::ui_layer, 400), tick);
    require(!accepts(layer, kind::ui_layer) && accepts(layer, kind::ui_alpha) &&
        layer.counters()[ui_counter::trust_revoked_declared] == 1, "An accepted UIAlpha did not judge the layer");
    alpha_auto_policy judge;
    judge.restore("ui_layer:10:srgb,backbuffer:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 1000)
      feed(judge, sample().alpha(kind::ui_layer, 20).alpha(kind::backbuffer, 400), tick);
    require(accepts(judge, kind::backbuffer) && accepts(judge, kind::ui_layer), "The inferred layer judged presented alpha");
    // An invalid or unaccepted declared alpha judges nothing: a full UIAlpha
    // is never selective, so never accepted.
    alpha_auto_policy alone;
    alone.restore("current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 8000; tick += 1000) {
      feed(alone, sample().alpha(kind::ui_alpha, 1000).alpha(kind::current, 400), tick);
      feed(alone, sample().alpha(kind::ui_color, 20, 600).alpha(kind::current, 400), tick + 500);
    }
    require(accepts(alone, kind::current) && !accepts(alone, kind::ui_alpha),
      "An unaccepted or invalid declared alpha revoked presented alpha");
    // Declared sources are never judged: an accepted UIAlpha beside a valid
    // exact pair and disagreeing presented alpha stays accepted.
    alpha_auto_policy declared;
    declared.restore("ui_alpha:61:srgb");
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 500)
      feed(declared, sample().alpha(kind::ui_alpha, 1000).alpha(kind::current, 100).pair(100, 900), tick);
    require(accepts(declared, kind::ui_alpha), "A declared source was judged");
  }

  // A HUD-less change set is declared: accepted by its first selective
  // sample from an exact pair; an inexact pair never earns.
  void exact_change_sets_earn() {
    alpha_auto_policy policy;
    for (std::uint64_t tick = 1000; tick <= 10000; tick += 100) feed(policy, sample().pair(50, 900, false), tick);
    feed(policy, sample().pair(980, 10), 10100);                // A full change set (a menu).
    feed(policy, sample().pair(50, 900, true, 100), 10200);     // Too few matching tiles.
    feed(policy, sample().pair(50, 900).alpha(kind::ui_alpha, 0, 20), 10300); // Void.
    require(!accepts(policy, kind::hudless), "An inexact, full, noisy or void pair was accepted");
    feed(policy, sample().pair(50, 900), 10400);
    require(accepts(policy, kind::hudless) && policy.accepted(candidate::hudless | candidate::exact, signatures()) ==
        candidate::hudless, "An exact selective change set was not accepted");
  }

  // S2: manual On accepts the offered candidates for this session only; it
  // never earns, revokes or persists. Manual Off accepts nothing.
  void manual_on_is_a_session_override() {
    alpha_auto_policy policy;
    std::vector<std::string> heard;
    policy.on_change([&](const std::string &stored) { heard.push_back(stored); });
    policy.restore("backbuffer:24:srgb");
    policy.set_manual(true);
    require(policy.accepted(0x7fu, signatures()) == (0x7fu & ui_selection::candidate_bits) &&
        policy.accepted(0x17fu, signatures()) == ui_selection::candidate_bits && !policy.accepted(0x20u, signatures()),
      "Manual On did not accept exactly the offered candidates");
    for (std::uint64_t tick = 1000; tick <= 70000; tick += 500) {
      feed(policy, sample().alpha(kind::current, 200).alpha(kind::ui_color, 30).pair(50, 900), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900).one_way(kind::backbuffer, 1000, 800), tick);
    }
    policy.set_manual(false);
    require(!policy.accepted(0x7fu, signatures()), "Manual Off accepted a candidate");
    policy.set_automatic();
    require(policy.stored() == "backbuffer:24:srgb" && heard.empty() && !policy.counters()[ui_counter::trust_earned] &&
        !policy.counters()[ui_counter::trust_revoked_exact] && !policy.counters()[ui_counter::trust_lapsed],
      "Manual On earned, revoked, lapsed or persisted");
    require(policy.accepted(0x7fu, signatures()) == candidate::backbuffer, "Auto lost the ledger after manual On");
  }

  // F1: the status is keyed on (scope, winner): the first offered and
  // accepted candidate in draw order of the adopted inputs. A sample is
  // discarded only under another scope or stale on arrival; a change of the
  // other inputs keeps it, and the status shows it only under its winner.
  void status_follows_the_winner_and_scope() {
    namespace temporal = ui_temporal;
    temporal::detection_state state;
    require(!state.status_key(), "A state without adopted inputs had a winner");
    // Resident Evil Requiem with FG: an accepted UI color tag (2) and current
    // alpha (8); its HUD-less pair (16) joins on some Presents only.
    state.adopt(2u, 2u | 8u);
    const auto tag = state.status_key();
    state.adopt(2u | 8u | 16u, 2u | 8u);
    require(tag == candidate::ui_color && state.status_key() == tag, "A pairing or inferred alpha beside the tag moved the winner");
    state.adopt(2u | 48u, 2u | 8u | 16u);
    require(state.status_key() == candidate::ui_color, "An accepted HUD-less pair beside the tag won");
    // The first accepted candidate in draw order wins, whatever is offered
    // beside it; unaccepted candidates never key it, and stored flags are no
    // input of adopt.
    state.adopt(0x40u | 4u | 48u, 0x40u | 4u);
    require(state.status_key() == candidate::layer, "The accepted layer did not win before Backbuffer");
    state.adopt(0x40u | 4u, 4u);
    require(state.status_key() == candidate::backbuffer, "An unaccepted layer won");
    state.adopt(16u | 32u, 16u);
    require(state.status_key() == candidate::hudless, "An accepted HUD-less pair did not win alone");
    state.adopt(2u | 4u | 16u, 0u);
    require(!state.status_key(), "Nothing accepted had a winner");
    // Adoption keeps the latest sample: status_fresh decides what it describes.
    alpha_auto_source scope;
    scope.epoch = 3; scope.revision = 4; scope.viewport = 5; scope.now_ms = 1200;
    state.adopt(2u, 2u);
    state.latest.sample_tick_ms = 1000;
    state.latest_source = scope;
    state.latest_key = candidate::ui_color;
    state.adopt(2u | 8u | 16u, 2u | 8u);
    require(state.latest.sample_tick_ms == 1000 && state.status_fresh(scope), "A pairing beside the winner dropped or staled the sample");
    state.adopt(8u, 8u);
    require(state.latest.sample_tick_ms == 1000 && !state.status_fresh(scope), "A sample described another winner");
    state.adopt(2u, 2u);
    auto later = scope;
    later.now_ms = 1501;
    require(state.status_fresh(scope) && !state.status_fresh(later), "The status freshness bound moved");
    auto other = scope;
    ++other.revision;
    require(!state.status_fresh(other), "A sample described another scope");
    // A sample describes frames within 500 ms of its tick, and is discarded
    // only under another scope or when stale on arrival.
    require(!temporal::sample_stale(1000, 1500) && temporal::sample_stale(1000, 1501) && temporal::sample_stale(0, 10),
      "The sample freshness bound moved");
    alpha_auto_source now, pending;
    now.now_ms = 1200; pending.now_ms = 1000;
    require(!temporal::sample_discarded(now, pending), "A sample in scope was discarded");
    now.now_ms = 1501;
    require(temporal::sample_discarded(now, pending), "A sample stale on arrival was kept");
    now.now_ms = 999;
    require(temporal::sample_discarded(now, pending), "A sample from the future was kept");
    now.now_ms = 1200;
    ++now.viewport;
    require(temporal::sample_discarded(now, pending), "A sample from another viewport was kept");
  }

  // A sample that decided H1 (source 8) covers the whole frame with an
  // unaccepted candidate: its counts are never selective, so it never earns;
  // without a visible scene it revokes nothing. The first-run shadow is its
  // owner's toggle.
  void samples_that_flatten_learn_nothing() {
    require(ui_detection::scene_visible(.25f) && !ui_detection::scene_visible(.2499f), "The visible bound moved");
    // The full-frame counts of an H1 sample never earn.
    alpha_auto_policy policy;
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 100)
      feed(policy, sample().alpha(kind::ui_layer, pixels).pair(pixels, 0), tick);
    for (std::uint64_t tick = 7000; tick <= 12000; tick += 100)
      feed(policy, sample().alpha(kind::ui_alpha, pixels).alpha(kind::current, pixels).pair(pixels, 0), tick);
    require(policy.stored().empty(), "A full-frame hidden-scene sample earned acceptance");
    // A full sample without a visible scene revokes nothing.
    alpha_auto_policy accepting;
    accepting.restore("ui_layer:10:srgb");
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 100)
      feed(accepting, sample().alpha(kind::ui_layer, pixels).pair(pixels, 0, false), tick);
    require(accepts(accepting, kind::ui_layer), "A full-frame sample without a visible scene revoked acceptance");
    // The first-run shadow is its owner's toggle (UISceneShadow), independent
    // of acceptance: restoring, earning, revoking or forgetting never changes it.
    alpha_auto_policy first;
    require(!first.first_run(), "A policy started as a first run without its owner saying so");
    first.restore("ui_layer:10:srgb");
    require(!first.first_run(), "Restored acceptance set the first-run shadow");
    first.set_first_run(true);
    feed(first, sample().alpha(kind::ui_alpha, 100), 1000);
    require(first.first_run() && accepts(first, kind::ui_alpha), "Earning acceptance ended the first-run shadow");
    first.forget();
    require(first.first_run(), "Forget ended the first-run shadow");
  }

  // T1 (game3d_ui_temporal.h): a generated Present shows the decision of the
  // real frame it shows, within the tag bound and the scope, else no mask; a
  // real Present always detects and pushes its per-frame bits. The runtime
  // tests prove the renderer calls it; these pin its rules without a GPU.
  void temporal_state_holds() {
    namespace temporal = ui_temporal;
    using temporal::hold_kind;
    using temporal::present_identity;
    alpha_auto_source scope;
    scope.epoch = 1; scope.revision = 2; scope.viewport = 3;
    temporal::detection_state state;
    // Nothing decided yet: a generated Present has no mask, a real one
    // detects from a new chain.
    const auto none = state.arbitrate(present_identity{true, 5}, scope, 0u);
    require(none.kind == hold_kind::unavailable && !none.hold && !none.detect && !none.adopt && !none.per_frame,
      "A generated Present held a mask no decision made");
    const auto first = state.arbitrate(present_identity{false, 5}, scope, 16u | 32u);
    require(first.detect && !first.hold && first.kind == hold_kind::none && first.adopt &&
        first.per_frame == ui_detection::per_frame_hold_reset, "The first real Present did not detect from a new chain");
    state.adopt(16u | 32u, 16u);
    state.detected(scope, present_identity{false, 5}, 16u | 32u);
    // Generated Presents of that real frame and of the next one hold, as many
    // as come (no multiplier constant); a third real frame does not.
    for (std::uint32_t i = 0; i != 5; ++i) {
      const auto hold = state.arbitrate(present_identity{true, 5}, scope, 0u);
      require(hold.hold && hold.kind == hold_kind::generated && !hold.detect && !hold.adopt && !hold.per_frame,
        "A generated Present of the decided real frame did not hold");
      state.held(present_identity{true, 5});
    }
    require(state.holds == 5 && !state.next_frame, "Holds of the decided real frame set the next frame");
    require(state.arbitrate(present_identity{true, 6}, scope, 0u).hold, "A generated Present of the next real frame did not hold");
    state.held(present_identity{true, 6});
    require(state.next_frame == 6 && state.arbitrate(present_identity{true, 5}, scope, 0u).hold &&
        state.arbitrate(present_identity{true, 6}, scope, 0u).hold, "The tag bound lost the decided or the next real frame");
    const auto beyond = state.arbitrate(present_identity{true, 7}, scope, 0u);
    require(!beyond.hold && beyond.kind == hold_kind::unavailable, "A generated Present beyond the next real frame held");
    require(state.arbitrate(present_identity{true, 0}, scope, 0u).hold, "A generated Present without a HUD-less tag did not hold");
    // A generated Present under another identity scope has no mask, and a
    // real one there starts a new chain.
    auto recreated = scope;
    ++recreated.epoch;
    require(state.arbitrate(present_identity{true, 5}, recreated, 0u).kind == hold_kind::unavailable &&
        state.arbitrate(present_identity{false, 6}, recreated, 48u).per_frame == ui_detection::per_frame_hold_reset,
      "A hold crossed an epoch change");
    // An observation revision alone (a depth observation loss) is a missing
    // input, not another identity: a generated Present holds, and a real one
    // missing the accepted pair reuses the previous decision once (T1).
    auto lost = scope;
    ++lost.revision;
    require(state.arbitrate(present_identity{true, 5}, lost, 0u).hold &&
        state.arbitrate(present_identity{false, 6}, lost, 0u).per_frame == ui_detection::per_frame_accepted_missing,
      "An observation revision reset the hold chain");
    auto resized = scope;
    ++resized.viewport;
    require(!state.arbitrate(present_identity{true, 5}, resized, 0u).hold &&
        state.arbitrate(present_identity{false, 6}, resized, 48u).per_frame == ui_detection::per_frame_hold_reset,
      "A hold crossed a viewport change");
    // The next real frame in scope detects without per-frame bits and adopts.
    const auto next = state.arbitrate(present_identity{false, 6}, scope, 48u);
    require(next.detect && next.adopt && !next.per_frame, "A real frame in the chain pushed per-frame bits");
    state.detected(scope, present_identity{false, 6}, 48u);
    require(!state.holds && !state.next_frame && state.decision_frame == 6, "A real decision kept the hold count or next frame");
    // A Present without a mask ends the chain: generated Presents then have
    // no mask, and the next real detection starts a new chain.
    state.unavailable();
    require(state.arbitrate(present_identity{true, 6}, scope, 0u).kind == hold_kind::unavailable &&
        state.arbitrate(present_identity{false, 7}, scope, 48u).per_frame == ui_detection::per_frame_hold_reset,
      "A Present without a mask kept the chain");
    // An accepted candidate the last adopting real frame offered and this one
    // misses: accepted_missing, and the frame adopts nothing. Acceptance is
    // never inherited: an unaccepted current alpha offered instead changes
    // nothing, and an unaccepted missing candidate is no such frame.
    temporal::detection_state accepted;
    accepted.adopt(4u, 4u);
    accepted.detected(scope, present_identity{}, 4u);
    const auto missing = accepted.arbitrate(present_identity{}, scope, 0u);
    require(missing.detect && !missing.adopt && missing.per_frame == ui_detection::per_frame_accepted_missing,
      "A missing accepted candidate was not flagged, or the frame adopted");
    require(accepted.arbitrate(present_identity{}, scope, 8u).per_frame == ui_detection::per_frame_accepted_missing,
      "An unaccepted current alpha stood in for the missing Backbuffer");
    require(!accepted.arbitrate(present_identity{}, scope, 4u | 8u).per_frame && accepted.arbitrate(present_identity{}, scope, 4u).adopt,
      "An offered accepted candidate was flagged missing");
    temporal::detection_state unaccepted;
    unaccepted.adopt(4u | 0x40u, 0x40u);
    unaccepted.detected(scope, present_identity{}, 4u | 0x40u);
    require(!unaccepted.arbitrate(present_identity{}, scope, 0x40u).per_frame &&
        unaccepted.arbitrate(present_identity{}, scope, 4u).per_frame == ui_detection::per_frame_accepted_missing,
      "Missing candidates were flagged by the wrong acceptance");
    // A generated Present never flags or adopts, whatever it offers.
    const auto generated = accepted.arbitrate(present_identity{true, 0}, scope, 1u);
    require(generated.hold && !generated.per_frame && !generated.adopt, "A generated Present detected or adopted");
    // The pre-UI change set (fix 3) is offered only with an exact retained
    // pairing, so it is left out of the accepted-missing test: a real frame
    // without it adopts, and only the first one after a real frame that
    // offered it reports the change-set gap (b2 word 5), once.
    {
      const std::uint32_t pre_ui = ui_detection::candidate::pre_ui, layer = ui_detection::candidate::layer;
      temporal::detection_state menu;
      require(!menu.arbitrate(present_identity{}, scope, 8u | layer | pre_ui).change_set_gap,
        "A new chain reported a change-set gap");
      menu.adopt(8u | layer | pre_ui, 8u | layer | pre_ui);
      menu.detected(scope, present_identity{}, 8u | layer | pre_ui);
      const auto gap = menu.arbitrate(present_identity{}, scope, 8u | layer);
      require(gap.detect && gap.adopt && !gap.per_frame && gap.change_set_gap,
        "A frame without the pre-UI change set was flagged accepted_missing, did not adopt, or reported no gap");
      menu.adopt(8u | layer, 8u | layer);
      menu.detected(scope, present_identity{}, 8u | layer);
      const auto after = menu.arbitrate(present_identity{}, scope, 2u | 8u | layer);
      require(after.adopt && !after.per_frame && !after.change_set_gap,
        "The gap lasted past one frame, or the adopted inputs stopped updating");
      temporal::detection_state held;
      held.adopt(8u | pre_ui, 8u | pre_ui);
      held.detected(scope, present_identity{}, 8u | pre_ui);
      require(held.arbitrate(present_identity{true, 0}, scope, 0u).hold &&
          !held.arbitrate(present_identity{true, 0}, scope, 0u).change_set_gap &&
          held.arbitrate(present_identity{}, scope, 8u).change_set_gap,
        "A generated Present reported the gap, or the next real frame did not");
      held.unavailable();
      require(!held.arbitrate(present_identity{}, scope, 8u).change_set_gap, "A new chain inherited the change-set gap");
    }

    // A scope change discards the latest sample; an inactive render ends the
    // T1 chain. Neither holds any hidden-scene state (game3d_scene_guard.h).
    temporal::detection_state scoped;
    scoped.latest.sample_tick_ms = 1000;
    scoped.latest_source = scope;
    scoped.enter_scope(scope);
    require(scoped.latest.sample_tick_ms == 1000, "A detection in scope discarded the sample");
    auto moved = scope;
    ++moved.revision;
    scoped.enter_scope(moved);
    require(!scoped.latest.sample_tick_ms, "A new scope kept the sample");
    scoped.detected(scope, present_identity{false, 9}, 0u);
    scoped.inactive();
    require(!scoped.have_decision && scoped.arbitrate(present_identity{true, 9}, scope, 0u).kind == hold_kind::unavailable &&
        scoped.arbitrate(present_identity{false, 10}, scope, 0u).per_frame == ui_detection::per_frame_hold_reset,
      "An inactive render kept a decision");
  }

  // The hidden-scene guard (M5, game3d_scene_guard.h): a sample whose
  // presented evidence reads `presented` (D .05 hidden, .2 ambiguous, .5
  // visible; none is invalid evidence) with the offered layer opaque-full
  // and claiming the frame.
  scene_guard::sample scene_sample(std::uint64_t tick, ui_detection::scene_verdict presented,
      std::uint32_t offered = candidate::layer, std::uint32_t claims = candidate::layer) {
    using ui_detection::scene_verdict;
    scene_guard::sample s;
    s.tick = tick;
    s.pixels = pixels;
    s.offered = offered;
    s.valid_bits = offered & ui_selection::candidate_bits;
    s.claims = claims;
    s.opaque.fill(pixels);
    s.complete = true;
    s.presented.n = 400;
    s.presented.decided = 600;
    s.presented.ran = true;
    s.presented.valid = presented != scene_verdict::none;
    s.presented.verdict = presented;
    s.presented.d = presented == scene_verdict::hidden ? .05f : presented == scene_verdict::visible ? .5f : .2f;
    return s;
  }
  // The same sample offering Stellar Blade SDR's cleared output target: a
  // V1-invalid layer (colour without alpha) whose pre-UI claim names it, with
  // the pre-UI image's D (negative: no pre-UI evidence).
  scene_guard::sample pre_ui_sample(std::uint64_t tick, ui_detection::scene_verdict presented, float pre_ui_d) {
    auto s = scene_sample(tick, presented, candidate::layer | candidate::current, ui_detection::claim_pre_ui);
    s.valid_bits = candidate::current;
    s.opaque.fill(0);
    s.pre_ui_image = ui_detection::pre_ui_image::layer;
    s.pre_ui.valid = pre_ui_d >= 0.f;
    s.pre_ui.ran = true;
    s.pre_ui.n = 400;
    s.pre_ui.d = pre_ui_d;
    s.pre_ui.verdict = s.pre_ui.valid ? ui_detection::scene_verdict_of(pre_ui_d) : ui_detection::scene_verdict::none;
    return s;
  }

  void scene_guard_holds_refutes_and_scopes() {
    using ui_detection::scene_verdict;
    constexpr auto H = scene_verdict::hidden, V = scene_verdict::visible, A = scene_verdict::ambiguous,
      none = scene_verdict::none;
    const auto sigs = signatures().by_kind();
    constexpr std::uint32_t hidden_bit = ui_detection::per_frame_scene_hidden, pre_ui_bit = ui_detection::per_frame_pre_ui_visible;
    const auto layer_refuted = candidate::layer << ui_detection::per_frame_refuted_shift;
    namespace word = ui_detection::decision_word;

    // Two valid hidden samples within hold_ms enter the hidden hold, which
    // lasts hold_ms from the latest; one sample enters nothing.
    scene_guard::state g;
    g.enter_scope(1, 1);
    auto observed = g.observe(scene_sample(1000, H), true, sigs);
    require(!observed.entered && !g.per_frame(1000, candidate::layer, sigs, false), "One hidden sample entered a hold");
    observed = g.observe(scene_sample(1100, H), true, sigs);
    require(observed.entered && g.per_frame(1100, candidate::layer, sigs, false) == hidden_bit &&
        g.per_frame(1600, candidate::layer, sigs, false) == hidden_bit && !g.per_frame(1601, candidate::layer, sigs, false),
      "Two hidden samples did not enter a hold of hold_ms");
    // Each further hidden sample renews it without entering again.
    observed = g.observe(scene_sample(1500, H), true, sigs);
    require(!observed.entered && g.per_frame(2000, candidate::layer, sigs, false) == hidden_bit && !g.per_frame(2001, candidate::layer, sigs, false),
      "A hidden sample did not renew the hold");
    // An ambiguous sample breaks no held hold; invalid evidence and evidence
    // that is not actionable change nothing.
    g.observe(scene_sample(1700, A), true, sigs);
    g.observe(scene_sample(1800, none), true, sigs);
    g.observe(scene_sample(1900, V), false, sigs);
    require(g.per_frame(2000, candidate::layer, sigs, false) == hidden_bit && !g.refuted_count,
      "An ambiguous, invalid or shadow-only sample released or refuted");
    // One valid visible sample releases the hold and refutes the signature of
    // every candidate whose full claim it carried.
    observed = g.observe(scene_sample(1950, V), true, sigs);
    require(observed.released && observed.refuted == 1 && !g.per_frame(1950, 0u, sigs, false) &&
        g.refuted(signatures().of(kind::ui_layer)) && g.per_frame(1950, candidate::layer, sigs, false) == layer_refuted,
      "A visible sample did not release the hold and refute the layer");
    observed = g.observe(scene_sample(2000, V), true, sigs);
    require(!observed.released && !observed.refuted && g.refuted_count == 1, "A refuted signature was refuted again");

    // Beyond hold_ms apart two hidden samples enter nothing; an ambiguous one
    // breaks the entry run, an invalid one does not.
    scene_guard::state far;
    far.enter_scope(1, 1);
    far.observe(scene_sample(1000, H), true, sigs);
    require(!far.observe(scene_sample(1501, H), true, sigs).entered, "Hidden samples 501 ms apart entered a hold");
    require(far.observe(scene_sample(1600, H), true, sigs).entered, "Hidden samples 99 ms apart did not enter");
    scene_guard::state broken;
    broken.enter_scope(1, 1);
    broken.observe(scene_sample(1000, H), true, sigs);
    broken.observe(scene_sample(1100, A), true, sigs);
    require(!broken.observe(scene_sample(1200, H), true, sigs).entered, "An ambiguous sample did not break the entry run");
    require(broken.observe(scene_sample(1300, H), true, sigs).entered, "Hidden samples after a break did not enter");
    scene_guard::state gap;
    gap.enter_scope(1, 1);
    gap.observe(scene_sample(1000, H), true, sigs);
    gap.observe(scene_sample(1100, none), true, sigs);
    require(gap.observe(scene_sample(1200, H), true, sigs).entered, "Invalid evidence broke the entry run");
    scene_guard::state shadow_only;
    shadow_only.enter_scope(1, 1);
    shadow_only.observe(scene_sample(1000, H), false, sigs);
    shadow_only.observe(scene_sample(1100, H), false, sigs);
    require(!shadow_only.per_frame(1100, candidate::layer, sigs, false), "Evidence that was not actionable entered a hold");
    // A visible sample that decided H1 counts as a release even without a
    // hold at its tick (samples in flight).
    scene_guard::state late;
    late.enter_scope(1, 1);
    auto decided_visible = scene_sample(1000, V);
    decided_visible.source = 8;
    require(late.observe(decided_visible, true, sigs).released, "A visible H1 sample was not counted as a release");

    // The pre-UI hold: the samples that hit the hidden hold while carrying
    // the pre-UI claim read the pre-UI image visible (Stellar Blade SDR
    // settings: presented .031, layer .588). The layer's claim exists only
    // while the acceptance ledger proves its signature (layer_proven); the
    // guard neither gives nor withdraws that proof.
    scene_guard::state sb;
    sb.enter_scope(1, 1);
    const std::uint32_t sb_offer = candidate::layer | candidate::current;
    constexpr std::uint32_t proven_bit = ui_detection::per_frame_pre_ui_proven;
    sb.observe(pre_ui_sample(1000, H, .588f), true, sigs);
    require(sb.per_frame(1000, sb_offer, sigs, true) == proven_bit, "One pre-UI sample held");
    sb.observe(pre_ui_sample(1100, H, .588f), true, sigs);
    require(sb.per_frame(1100, sb_offer, sigs, true) == (hidden_bit | pre_ui_bit | proven_bit) &&
        sb.per_frame(1100, sb_offer, sigs, false) == hidden_bit,
      "Two pre-UI samples did not hold both verdicts for a proven layer alone");
    // The proven bit names an offered layer only, whatever the pre-UI image.
    require(sb.per_frame(1100, candidate::current, sigs, true) == (hidden_bit | pre_ui_bit) &&
        sb.per_frame(1100, candidate::layer | candidate::hudless, sigs, true) == (hidden_bit | pre_ui_bit | proven_bit),
      "The proven bit was pushed without an offered layer");
    // Absent pre-UI evidence, or evidence of another image, changes nothing.
    sb.observe(pre_ui_sample(1200, H, -1.f), true, sigs);
    auto other_image = pre_ui_sample(1300, H, .05f);
    other_image.pre_ui_image = ui_detection::pre_ui_image::hudless;
    sb.observe(other_image, true, sigs);
    require(sb.per_frame(1600, sb_offer, sigs, true) == (hidden_bit | pre_ui_bit | proven_bit),
      "Absent or mismatched pre-UI evidence changed a hold");
    // A pre-UI image that does not read visible releases the pre-UI hold only.
    sb.observe(pre_ui_sample(1400, H, .2f), true, sigs);
    require(sb.per_frame(1400, sb_offer, sigs, true) == (hidden_bit | proven_bit), "A non-visible pre-UI image kept the pre-UI hold");
    // Leaving the menu: the presented frame reads visible and releases both;
    // the pre-UI claim is never refuted.
    observed = sb.observe(pre_ui_sample(1500, V, .5f), true, sigs);
    require(observed.released && !observed.refuted && sb.per_frame(1500, sb_offer, sigs, true) == proven_bit && !sb.refuted_count,
      "Leaving the menu did not release both holds, or refuted the pre-UI claim");
    // Dark or foggy gameplay: both images read hidden, so the pre-UI hold
    // never enters (the hidden hold alone makes no claim act).
    scene_guard::state dark;
    dark.enter_scope(1, 1);
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 100) dark.observe(pre_ui_sample(tick, H, .05f), true, sigs);
    require(dark.per_frame(3000, sb_offer, sigs, true) == (hidden_bit | proven_bit), "A dark scene entered the pre-UI hold");
    // Gameplay: both images visible, nothing held.
    scene_guard::state gameplay;
    gameplay.enter_scope(1, 1);
    for (std::uint64_t tick = 1000; tick <= 2000; tick += 100) gameplay.observe(pre_ui_sample(tick, V, .518f), true, sigs);
    require(gameplay.per_frame(2000, sb_offer, sigs, true) == proven_bit, "Visible gameplay held a verdict");

    // An unproven layer makes no pre-UI claim (the GPU's claim (d) needs the
    // proven bit), so its samples never enter the pre-UI hold, and a pre-UI
    // hold held for the layer is not pushed while it is unproven.
    scene_guard::state unproven;
    unproven.enter_scope(1, 1);
    for (std::uint64_t tick = 1000; tick <= 1300; tick += 100) {
      auto no_claim = pre_ui_sample(tick, H, .588f);
      no_claim.claims = 0;
      unproven.observe(no_claim, unproven.measure(tick, false, false, false).actionable, sigs);
    }
    require(!unproven.per_frame(1300, sb_offer, sigs, false) && !unproven.pre_ui.until && !unproven.hidden.until,
      "An unproven layer without a claim held a verdict");
    // A proven layer that is the offer's pre-UI image measures actionably
    // without a claim or a hold, so that the first hidden sample after an
    // identity change already counts; beside a HUD-less image it does not.
    scene_guard::state fresh_guard;
    fresh_guard.enter_scope(1, 1);
    const auto proven_measure = fresh_guard.measure(1000, false, false, true),
      unproven_measure = fresh_guard.measure(1000, false, false, false);
    require(proven_measure.run && proven_measure.actionable && !unproven_measure.run && !unproven_measure.actionable,
      "A proven pre-UI layer did not measure actionably");
    // The 07:20 Stellar Blade session: FG suspended for the menu changes the
    // depth provider's viewport, an identity change that clears every hold
    // but not the ledger's proof; the menu holds both verdicts at its second
    // hidden sample.
    scene_guard::state suspended;
    suspended.enter_scope(1, 1);
    for (std::uint64_t tick = 1000; tick <= 2000; tick += 100) suspended.observe(pre_ui_sample(tick, V, .36f), true, sigs);
    suspended.enter_scope(1, 0);
    // A fade-in sample: presented visible (0.453), the layer 0.538.
    auto fade = pre_ui_sample(2100, V, .538f);
    fade.presented.d = .453f;
    suspended.observe(fade, suspended.measure(2100, false, false, true).actionable, sigs);
    suspended.observe(pre_ui_sample(2200, H, .544f), suspended.measure(2200, false, false, true).actionable, sigs);
    require(suspended.per_frame(2200, sb_offer, sigs, true) == proven_bit, "The first hidden menu sample held");
    suspended.observe(pre_ui_sample(2300, H, .569f), suspended.measure(2300, false, false, true).actionable, sigs);
    require(suspended.per_frame(2300, sb_offer, sigs, true) == (hidden_bit | pre_ui_bit | proven_bit),
      "The second hidden menu sample after an identity change did not hold both verdicts");
    // The pre-UI hold is per offer, the proof per signature (the caller's
    // ledger query): an unproven signature's per-frame bits carry no pre-UI
    // verdict.
    auto other_layer = signatures();
    other_layer.set(kind::ui_layer, 28);
    require(suspended.per_frame(2300, sb_offer, other_layer.by_kind(), false) == hidden_bit,
      "The pre-UI hold acted for an unproven signature");

    // Restore: a refuted candidate offered, valid and below 99% opaque is an
    // overlay again; opaque-full, invalid, or a sample without texel 10 does
    // not restore.
    auto opaque_layer = scene_sample(2100, A);
    g.observe(opaque_layer, true, sigs);
    require(g.refuted(signatures().of(kind::ui_layer)), "An opaque-full layer restored its refuted signature");
    auto incomplete = scene_sample(2200, A, candidate::layer, 0u);
    incomplete.opaque.fill(0);
    incomplete.complete = false;
    g.observe(incomplete, true, sigs);
    auto invalid_layer = scene_sample(2300, A, candidate::layer, 0u);
    invalid_layer.opaque.fill(0);
    invalid_layer.valid_bits = 0;
    g.observe(invalid_layer, true, sigs);
    require(g.refuted(signatures().of(kind::ui_layer)), "An invalid layer or a sample without texel 10 restored the layer");
    auto overlay = scene_sample(2400, A, candidate::layer, 0u);
    overlay.opaque[ui_selection::alpha_index(kind::ui_layer)] = pixels / 2;
    g.observe(overlay, true, sigs);
    require(!g.refuted_count && !g.per_frame(2400, candidate::layer, sigs, false), "A layer shown as an overlay stayed refuted");
    // A HUD-less signature: refuted by a visible sample of its exact full
    // change set, restored by an exact pair without the full claim.
    scene_guard::state pair;
    pair.enter_scope(1, 1);
    const std::uint32_t exact_pair = candidate::hudless | candidate::exact;
    pair.observe(scene_sample(1000, V, exact_pair, candidate::hudless), true, sigs);
    require(pair.refuted(signatures().of(kind::hudless)) &&
        pair.per_frame(1000, exact_pair, sigs, false) == (candidate::hudless << ui_detection::per_frame_refuted_shift),
      "A visible full change set did not refute the HUD-less signature");
    pair.observe(scene_sample(1100, A, candidate::hudless, 0u), true, sigs);
    require(pair.refuted(signatures().of(kind::hudless)), "An inexact pair restored the HUD-less signature");
    pair.observe(scene_sample(1200, A, exact_pair, 0u), true, sigs);
    require(!pair.refuted_count, "A partial exact pair did not restore the HUD-less signature");
    // Refutations are per signature, a bounded FIFO.
    scene_guard::state many;
    many.enter_scope(1, 1);
    for (std::uint32_t format = 100; format != 109; ++format) {
      auto other = signatures();
      other.set(kind::ui_layer, format);
      many.observe(scene_sample(1000 + format, V), true, other.by_kind());
    }
    auto first = signatures(), last = signatures();
    first.set(kind::ui_layer, 100);
    last.set(kind::ui_layer, 108);
    require(many.refuted_count == scene_guard::state::max_refuted && !many.refuted(first.of(kind::ui_layer)) &&
        many.refuted(last.of(kind::ui_layer)) && !many.refuted(signatures().of(kind::ui_layer)),
      "Refutations were not a bounded FIFO per signature");

    // per_frame composes the held verdicts, the refuted offered candidates
    // and the proven offered layer only.
    scene_guard::state composed;
    composed.enter_scope(1, 1);
    composed.observe(scene_sample(1000, V, candidate::layer | candidate::backbuffer, candidate::layer | candidate::backbuffer), true, sigs);
    composed.observe(pre_ui_sample(1050, V, .5f), true, sigs);
    composed.observe(pre_ui_sample(1100, H, .6f), true, sigs);
    composed.observe(pre_ui_sample(1200, H, .6f), true, sigs);
    require(composed.per_frame(1200, candidate::layer, sigs, true) == (hidden_bit | pre_ui_bit | layer_refuted | proven_bit) &&
        composed.per_frame(1200, candidate::layer, sigs, false) == (hidden_bit | layer_refuted) &&
        composed.per_frame(1200, candidate::backbuffer | candidate::current, sigs, true) ==
          (hidden_bit | pre_ui_bit | (candidate::backbuffer << ui_detection::per_frame_refuted_shift)) &&
        composed.per_frame(1200, 0u, sigs, false) == (hidden_bit | pre_ui_bit),
      "The per-frame bits were not the held verdicts, the refuted offered candidates and the proven layer");

    // Measurement: nothing runs without an acting-capable claim, a held
    // verdict, a proven pre-UI layer (above), the shadow or the whole-frame
    // diagnostic; only the first three are actionable.
    scene_guard::state measured;
    measured.enter_scope(1, 1);
    const auto idle = measured.measure(1000, false, false, false), shadow = measured.measure(1000, true, false, false),
      whole = measured.measure(1000, false, true, false);
    require(!idle.run && !idle.actionable && shadow.run && !shadow.actionable && whole.run && !whole.actionable,
      "The shadow or the whole-frame diagnostic measured actionably, or nothing measured");
    measured.observe(scene_sample(1000, none), false, sigs);
    require(measured.gate_open && measured.measure(1100, false, false, false).run && measured.measure(1100, false, false, false).actionable,
      "A sample with a claim did not open the gate");
    measured.observe(scene_sample(1100, V), true, sigs);
    require(!measured.gate_open && !measured.measure(1200, false, false, false).run,
      "A claim refuted by the same sample left the gate open");
    measured.observe(scene_sample(1200, none, candidate::layer | candidate::current, ui_detection::claim_pre_ui), false, sigs);
    require(measured.gate_open, "The pre-UI claim did not open the gate");
    measured.observe(scene_sample(1300, none, candidate::layer, candidate::layer), false, sigs);
    require(!measured.gate_open, "A refuted claim opened the gate");
    require(composed.measure(1200, false, false, false).actionable, "A held verdict did not measure actionably");

    // Only an identity change (epoch or viewport) clears the guard; a scope
    // entered again with the same identity (an observation revision, an
    // inactive frame, acceptance changes) keeps it.
    composed.enter_scope(1, 1);
    require(composed.per_frame(1200, candidate::layer, sigs, true) == (hidden_bit | pre_ui_bit | layer_refuted | proven_bit),
      "The same identity cleared the guard");
    auto epoch = composed, viewport = composed;
    epoch.enter_scope(2, 1);
    viewport.enter_scope(1, 2);
    require(!epoch.per_frame(1200, candidate::layer, sigs, false) && !epoch.refuted_count && !epoch.gate_open &&
        !viewport.per_frame(1200, candidate::layer, sigs, false) && !viewport.refuted_count,
      "An epoch or viewport change kept the guard's state");

    // The first-run shadow's uncovered hidden run: consecutive valid hidden
    // samples while no source decided and not blank, measured or not.
    scene_guard::state run;
    run.enter_scope(1, 1);
    require(!run.observe(scene_sample(1000, H, 0u, 0u), false, sigs).shadow_hidden_ms &&
        run.observe(scene_sample(1100, H, 0u, 0u), false, sigs).shadow_hidden_ms == 100 &&
        run.observe(scene_sample(1600, H, 0u, 0u), true, sigs).shadow_hidden_ms == 600,
      "The first-run shadow did not measure its uncovered hidden run");
    auto covered = scene_sample(1700, H, 0u, 0u);
    covered.source = 8;
    require(!run.observe(covered, false, sigs).shadow_hidden_ms && !run.observe(scene_sample(1800, H, 0u, 0u), false, sigs).shadow_hidden_ms,
      "A decided sample did not end the uncovered run");
    auto blank = scene_sample(1900, H, 0u, 0u);
    blank.presented.decided = ui_detection::scene::min_edges - 1;
    require(!run.observe(blank, false, sigs).shadow_hidden_ms && !run.observe(scene_sample(2000, V, 0u, 0u), false, sigs).shadow_hidden_ms,
      "A blank or visible sample extended the uncovered run");

    // sample_of decodes the decision words the guard reads: texels 0, 1, 4, 5,
    // 6, 7 and 10.
    std::array<std::uint32_t, 4 * ui_detection::h1_decision_texels> t{};
    t[word::source] = 8; t[word::pixels] = pixels; t[word::candidates] = 0x48;
    t[word::alpha_opaque] = 11; t[word::alpha_opaque + 1] = 12; t[word::layer_opaque] = 13; t[word::valid_bits] = 0x8;
    t[word::opaque_backbuffer] = 14; t[word::opaque_current] = 15; t[word::claims] = 0x80;
    t[word::h1] = 0u | ui_detection::h1_applied;
    const float d = .031f, pre_ui_d = .588f;
    t[word::scene_n] = 400; std::memcpy(&t[word::scene_d], &d, sizeof(d));
    t[word::scene_state] = 1u | 2u | std::uint32_t(H) << 2; t[word::scene_decided] = 600;
    t[word::pre_ui_scene_n] = 380; std::memcpy(&t[word::pre_ui_scene_d], &pre_ui_d, sizeof(pre_ui_d));
    t[word::pre_ui_scene_state] = 3u; t[word::pre_ui_scene_image] = ui_detection::pre_ui_image::layer;
    const auto decoded = scene_guard::sample_of(t.data(), t.size(), 1234, true);
    require(decoded.tick == 1234 && decoded.source == 8 && decoded.pixels == pixels && decoded.offered == 0x48 &&
        decoded.valid_bits == 0x8 && decoded.claims == 0x80 && decoded.complete &&
        decoded.opaque == std::array<std::uint32_t, 5>{11, 12, 13, 14, 15} && decoded.presented.valid &&
        decoded.presented.verdict == H && decoded.presented.decided == 600 && decoded.pre_ui.valid && decoded.pre_ui.n == 380 &&
        decoded.pre_ui.verdict == V && decoded.pre_ui_image == ui_detection::pre_ui_image::layer,
      "The scene guard did not decode a sample");
    const auto older = scene_guard::sample_of(t.data(), 4 * ui_detection::judgment_decision_texels, 1234, false);
    require(!older.complete && !older.claims && !older.presented.valid && !older.pre_ui_image,
      "The scene guard decoded texel 10 or evidence it was not given");

    // H2 (game3d_still_screen.h): the guard feeds its run from the samples it
    // observes in scope; acceptance, holds and refutations never clear it,
    // an identity change does; measure(still) runs without being actionable.
    scene_guard::state still;
    still.enter_scope(1, 1);
    const auto still_sample = [&](std::uint64_t tick) {
      auto s = scene_sample(tick, H, 0u, 0u);
      s.presented.d = -.03f;
      s.still_cells = s.still_compared = still_screen::cells;
      return s;
    };
    for (std::uint64_t tick = 1000; tick < 3000; tick += 100)
      require(!still.observe(still_sample(tick), false, sigs, true).still.entered, "H2 entered before 2 s");
    require(still.observe(still_sample(3000), false, sigs, true).still.entered && still.still_flatten(),
      "The guard's H2 run did not enter at 2 s of still hidden samples");
    require(!still.observe(still_sample(3100), true, sigs, true).still.ended && still.still_flatten() &&
        !still.enter_scope(1, 1).ended && still.still_flatten(), "An actionable sample or the same scope ended H2's run");
    const auto still_idle = still.measure(3100, false, false, false), still_measured = still.measure(3100, false, false, false, true);
    require(!still_idle.run && still_measured.run && !still_measured.actionable, "measure(still) was not a non-actionable run");
    auto still_out = still_sample(3200);
    require(still.observe(still_out, false, sigs, false).still.ended && !still.still_flatten(),
      "A sample submitted out of H2's scope did not end its run");
    for (std::uint64_t tick = 4000; tick <= 6000; tick += 100) still.observe(still_sample(tick), false, sigs, true);
    const auto ended = still.enter_scope(2, 1);
    require(ended.ended && ended.reason == still_screen::end_reason::identity && !still.still_flatten(),
      "An identity change did not end H2's run");
    // Evidence measured for H2 alone never reaches the acceptance ledger: its
    // scene evidence is cleared, so a pre-UI proof's reconfirm clock stays
    // paused; evidence measured for any other rule passes unchanged.
    {
      alpha_auto_decision::detection_evidence measured;
      measured.scene = {300u, 600u, .4f, true, true, ui_detection::scene_verdict::visible};
      measured.pre_ui_scene = measured.scene;
      measured.pre_ui_match = 7u;
      const auto h2_only = ui_temporal::ledger_evidence(measured, true), other = ui_temporal::ledger_evidence(measured, false);
      require(!h2_only.scene.valid && !h2_only.scene.ran && !h2_only.pre_ui_scene.valid && h2_only.pre_ui_match == 7u &&
          other.scene.valid && other.scene.verdict == ui_detection::scene_verdict::visible, "ledger_evidence did not hide only H2's measurement");
    }
  }
  // H1 (d), fix 1: Stellar Blade SDR's cleared output target, an offered
  // layer without coverage (BGRA8, V1-invalid: colour beyond the
  // premultiplied bound) and its pre-UI pixel counts against the presented
  // frame (decision texel 11), under the presented frame's D verdict.
  candidate_signatures sb_signatures(std::uint32_t format = 87, std::uint32_t color_space = 1) {
    candidate_signatures result;
    result.color_space = color_space;
    result.set(kind::ui_layer, format).set(kind::current, 24);
    return result;
  }
  alpha_auto_decision::detection_evidence sb_layer(std::uint32_t match, std::uint32_t lit,
      ui_detection::scene_verdict presented = ui_detection::scene_verdict::visible, std::uint32_t covered = 0) {
    alpha_auto_decision::detection_evidence e;
    e.candidates = candidate::layer | candidate::current;
    e.layer_covered = covered;
    e.layer_invalid = pixels;
    e.valid_bits = candidate::current;
    e.pre_ui_match = match;
    e.pre_ui_image_lit = lit;
    e.presented_lit = lit;
    e.presented_lit_differs = pixels - match;
    e.scene.valid = presented != ui_detection::scene_verdict::none;
    e.scene.ran = true;
    e.scene.verdict = presented;
    e.scene.n = 400;
    e.scene.d = presented == ui_detection::scene_verdict::hidden ? .02f : .4f;
    return e;
  }

  // The ledger's pre-UI proof (H1 d): earned like an inferred source by
  // matching samples, never withdrawn by a mismatch, keyed by the layer's
  // format and colour space, remembered, lapsing only over testable time,
  // cleared by Forget, and never a candidate's acceptance.
  void pre_ui_proof_is_earned_by_pixels() {
    using ui_detection::scene_verdict;
    namespace n = ui_counter;
    const auto sb = sb_signatures();
    const auto layer = sb.of(kind::ui_layer);
    const auto match = sb_layer(995, 900), mismatch = sb_layer(350, 900), menu = sb_layer(330, 260, scene_verdict::hidden);
    require(ui_selection::pre_ui_key(layer).key() == "pre_ui:87:srgb", "The pre-UI key text changed");

    // Three matching samples over 2 s earn it; two over 2 s do not.
    alpha_auto_policy earning;
    std::string heard;
    earning.on_change([&](const std::string &stored) { heard = stored; });
    earning.observe(match, pixels, 1000, sb);
    earning.observe(match, pixels, 3000, sb);
    require(!earning.pre_ui_proven(layer), "Two matching samples over 2 s proved the layer");
    earning.observe(match, pixels, 3100, sb);
    require(earning.pre_ui_proven(layer) && earning.stored() == "pre_ui:87:srgb" && heard == "pre_ui:87:srgb" &&
        earning.counters()[n::trust_earned] == 1, "Three matching samples over 2 s did not prove the layer");
    alpha_auto_policy quick;
    for (std::uint64_t tick = 1000; tick <= 2900; tick += 100) quick.observe(match, pixels, tick, sb);
    require(!quick.pre_ui_proven(layer), "Matching samples within less than 2 s proved the layer");
    // Never a candidate's acceptance: no candidate bit, and the layer itself
    // stays unaccepted.
    require(!earning.accepted(candidate::layer | candidate::current, sb) && !earning.accepts(layer),
      "The pre-UI proof was accepted as a candidate");
    // A mismatch (UI over the scene, a menu) never withdraws it.
    for (std::uint64_t tick = 4000; tick <= 90000; tick += 500) earning.observe(tick % 1000 ? mismatch : menu, pixels, tick, sb);
    require(earning.pre_ui_proven(layer) && earning.counters()[n::trust_earned] == 1, "A mismatch withdrew the proof");
    // Nor does a mismatch restart the earning run.
    alpha_auto_policy interrupted;
    interrupted.observe(match, pixels, 1000, sb);
    interrupted.observe(mismatch, pixels, 1500, sb);
    interrupted.observe(match, pixels, 2000, sb);
    interrupted.observe(menu, pixels, 2500, sb);
    interrupted.observe(match, pixels, 3000, sb);
    require(interrupted.pre_ui_proven(layer), "A mismatch restarted the earning run");

    // A dark image (lit on less than half the pixels: a transparent black UI
    // layer equals a black frame) and a layer with coverage never earn.
    alpha_auto_policy dark, covered;
    for (std::uint64_t tick = 1000; tick <= 9000; tick += 500) {
      dark.observe(sb_layer(pixels, 499), pixels, tick, sb);
      covered.observe(sb_layer(pixels, pixels, scene_verdict::visible, 10), pixels, tick, sb);
    }
    require(!dark.pre_ui_proven(layer) && !covered.pre_ui_proven(layer) && dark.stored().empty() && covered.stored().empty(),
      "A dark image or a layer with coverage was proven");

    // Another format or colour space is another key.
    alpha_auto_policy keys;
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) keys.observe(match, pixels, tick, sb_signatures(87, 3));
    require(keys.pre_ui_proven(sb_signatures(87, 3).of(kind::ui_layer)) && !keys.pre_ui_proven(layer) &&
        !keys.pre_ui_proven(sb_signatures(28).of(kind::ui_layer)) && keys.stored() == "pre_ui:87:pq",
      "The proof was not keyed by format and colour space");

    // Remembered: the key round-trips through stored() and restore(); a
    // restored proof is provisional, confirmed by three matches over 2 s.
    alpha_auto_policy restored;
    require(restored.restore(earning.stored()).restored == 1 && restored.pre_ui_proven(layer) &&
        restored.stored() == "pre_ui:87:srgb" && !restored.pre_ui_proven(sb_signatures(87, 3).of(kind::ui_layer)),
      "The pre-UI key did not round-trip");
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) restored.observe(match, pixels, tick, sb);
    require(restored.counters()[n::trust_earned] == 1 && restored.counters()[n::trust_restored] == 1,
      "Matching samples did not confirm a restored proof");
    for (std::uint64_t tick = 4000; tick <= 200000; tick += 1000) restored.observe(mismatch, pixels, tick, sb);
    require(restored.pre_ui_proven(layer) && !restored.counters()[n::trust_lapsed], "A confirmed proof lapsed");

    // A provisional proof lapses after 60 s of testable time (layer offered
    // without coverage, presented evidence valid and visible) without a match.
    alpha_auto_policy lapsing;
    lapsing.restore("pre_ui:87:srgb");
    for (std::uint64_t tick = 1000; tick <= 60000; tick += 1000) lapsing.observe(mismatch, pixels, tick, sb);
    require(lapsing.pre_ui_proven(layer), "A provisional proof lapsed before 60 s of testable time");
    lapsing.observe(mismatch, pixels, 61000, sb);
    require(!lapsing.pre_ui_proven(layer) && lapsing.stored().empty() && lapsing.counters()[n::trust_lapsed] == 1,
      "A provisional proof did not lapse after 60 s of testable time");
    // The clock pauses on every sample that is not testable: the presented
    // frame hidden or its evidence invalid, the layer missing, covered or of
    // another signature.
    alpha_auto_policy paused;
    paused.restore("pre_ui:87:srgb");
    for (std::uint64_t tick = 1000; tick <= 30000; tick += 1000) paused.observe(mismatch, pixels, tick, sb);
    alpha_auto_decision::detection_evidence no_layer;
    no_layer.candidates = candidate::current;
    std::uint64_t tick = 31000;
    for (; tick <= 200000; tick += 1000) {
      switch ((tick / 1000) % 5) {
        case 0: paused.observe(menu, pixels, tick, sb); break;
        case 1: paused.observe(sb_layer(350, 900, scene_verdict::none), pixels, tick, sb); break;
        case 2: paused.observe(no_layer, pixels, tick, sb); break;
        case 3: paused.observe(sb_layer(350, 900, scene_verdict::visible, 10), pixels, tick, sb); break;
        default: paused.observe(mismatch, pixels, tick, sb_signatures(28)); break;
      }
    }
    require(paused.pre_ui_proven(layer), "A provisional proof lapsed while its clock was paused");
    // 30 s ran before the pause (1000 to the first paused sample at 31000);
    // 30 more testable seconds lapse it.
    for (tick = 201000; tick <= 230000; tick += 1000) paused.observe(mismatch, pixels, tick, sb);
    require(paused.pre_ui_proven(layer), "A paused clock was charged with untestable time");
    paused.observe(mismatch, pixels, 231000, sb);
    require(!paused.pre_ui_proven(layer), "A provisional proof did not lapse after 60 s of testable time in all");

    // Forget clears it and reports it; the next menu needs it earned again.
    std::string forgotten = "unset";
    earning.on_change([&](const std::string &stored) { forgotten = stored; });
    require(earning.forget() == "pre_ui:87:srgb" && forgotten.empty() && !earning.pre_ui_proven(layer) &&
        earning.counters()[n::trust_forgotten] == 1, "Forget did not clear the pre-UI proof");

    // Fix 3: the pre-UI change set (candidate 0x100) is accepted exactly
    // while the proof of its layer's signature (its signature's key) is
    // held, and offering it changes no ledger entry: observe() never reads it.
    {
      auto offered_sb = sb;
      offered_sb.set(kind::pre_ui, 87);
      alpha_auto_policy proven, unproven;
      for (std::uint64_t t = 1000; t <= 3000; t += 1000) proven.observe(match, pixels, t, sb);
      const std::uint32_t offered = candidate::layer | candidate::current | candidate::pre_ui;
      require(proven.pre_ui_proven(layer) && proven.accepted(offered, offered_sb) == candidate::pre_ui &&
          !unproven.accepted(offered, offered_sb) &&
          !proven.accepted(offered, sb_signatures(87, 3).set(kind::pre_ui, 87)),
        "The pre-UI change set was not accepted exactly while its layer's proof is held");
      // The same samples with and without the pre-UI change set offered (and
      // valid) leave the same ledger and acceptance counters.
      alpha_auto_policy with_bit, without_bit;
      std::uint64_t t = 1000;
      for (const auto &e : {match, mismatch, menu, match, match, mismatch, match}) {
        auto offered_e = e;
        offered_e.candidates |= candidate::pre_ui;
        offered_e.valid_bits |= candidate::pre_ui;
        with_bit.observe(offered_e, pixels, t, offered_sb);
        without_bit.observe(e, pixels, t, sb);
        t += 700;
      }
      require(with_bit.stored() == without_bit.stored() && with_bit.stored() == "pre_ui:87:srgb" &&
          with_bit.counters().value == without_bit.counters().value,
        "Offering the pre-UI change set changed a ledger entry");
    }

    // Manual modes never earn it, but honour it.
    alpha_auto_policy manual;
    manual.set_manual(true);
    for (std::uint64_t t = 1000; t <= 5000; t += 1000) manual.observe(match, pixels, t, sb);
    require(!manual.pre_ui_proven(layer), "A manual mode earned the pre-UI proof");
    manual.set_automatic();
    for (std::uint64_t t = 6000; t <= 8000; t += 1000) manual.observe(match, pixels, t, sb);
    manual.set_manual(true);
    require(manual.pre_ui_proven(layer), "A manual mode did not honour an earned proof");
  }
  // Exact UI counters (game3d_ui_counters.h): the log text, the accounting
  // identity, wrap-safe GPU deltas, session totals and acceptance events.
  void ui_counters_are_formatted_and_trust_events_counted() {
    namespace n = ui_counter;
    ui_counters c;
    c[n::auto_frames] = 10; c[n::detection_frames] = 6;
    c[n::held_generated] = 2; c[n::held_none] = 1; c[n::reused] = 1;
    c[n::inactive_no_candidates] = 1;
    c[n::decided + 0] = 2; c[n::decided + 5] = 3; c[n::decided + 6] = 1; c[n::decided + 10] = 4; c[n::decided + 11] = 5;
    c[n::none + ui_no_mask::difference_failed] = 1; c[n::none + ui_no_mask::gate_no_hold] = 1;
    c[n::none + ui_no_mask::unaccepted] = 2;
    c[n::depth_not_current] = 1; c[n::full_d_hidden] = 1; c[n::inexact_difference] = 3; c[n::trust_earned] = 1;
    c[n::trust_discarded] = 2; c[n::trust_revoked_exact] = 1; c[n::trust_forgotten] = 3; c[n::contradicted] = 2;
    c[n::full_alpha] = 2; c[n::full_alpha_d_visible] = 1;
    c[n::scene_entered] = 1; c[n::scene_released] = 1; c[n::scene_refuted] = 2;
    c[n::still_entered] = 2; c[n::still_released] = 1; c[n::still_short] = 7;
    c[n::decided + 12] = 6; c[n::refined] = 4;
    c[n::change_set_samples] = 9; c[n::change_set_retained] = 5; c[n::change_set_late] = 3; c[n::change_set_unavailable] = 1;
    c[n::change_set_valid] = 4; c[n::change_set_would_refine] = 3; c[n::change_set_would_decide] = 1;
    c[n::change_set_pair_verified] = 4; c[n::change_set_pair_contradicted] = 1;
    // Fix 4: three darkening samples, two of them unpinning.
    c.add_darkening(250, 40); c.add_darkening(0, 5); c.add_darkening(50, 0);
    c[n::samples] = 4; c.through_ms = 12345;
    c[n::detection_frames] += 5; c[n::auto_frames] += 5;
    require(c.reconciled() && c.held() == 3 && c.inactive() == 1, "The counters' accounting identity is wrong");
    // reused frames are detection frames, not holds.
    ++c[n::reused];
    require(c.reconciled(), "A reused frame changed the accounting identity");
    --c[n::reused];
    require(format_ui_counters(c) ==
        "auto_frames=15 detection_frames=11 held={generated=2 none=1} reused=1 "
        "inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 10=4 11=5 12=6} refined=4 "
        "none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 gate_no_hold=1 "
        "no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 depth_not_current=1} full_d={hidden=1 ambiguous=0 visible=0 invalid=0} "
        "scene={entered=1 released=1 refuted=2} still={entered=2 released=1 short=7} change_set={samples=9 retained=5 late=3 "
        "unavailable=1 valid=4 would_refine=3 would_decide=1 pair_verified=4 pair_contradicted=1} "
        "darkening={samples=3 unpinned_samples=2 unpinned_px=300 kept_px=45} untrusted_inferred=0 "
        "inexact_difference=3 contradicted=2 presented_over_dedicated=0 full_alpha=2 "
        "full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_exact=1 "
        "revoked_declared=0 lapsed=0 restored=0 discarded=2 forgotten=3} samples=4 through_ms=12345",
      "The UI counters log text changed");
    ++c[n::auto_frames];
    require(!c.reconciled(), "An unaccounted frame reconciled");
    // GPU words are uint32 totals: their change survives a wrap.
    std::array<std::uint32_t, ui_counter_word::count> before{}, now{};
    before[ui_counter_word::detection_frames] = 0xfffffffeu; now[ui_counter_word::detection_frames] = 1u;
    before[ui_counter_word::decided + 4] = 7u; now[ui_counter_word::decided + 4] = 9u;
    now[ui_counter_word::decided + 10] = 6u;
    now[ui_counter_word::none + ui_no_mask::ambiguous] = 5u; now[ui_counter_word::none + ui_no_mask::unaccepted] = 3u;
    now[ui_counter_word::untrusted_inferred] = 2u;
    now[ui_counter_word::full_alpha] = 4u;
    now[ui_counter_word::contradicted] = 7u; now[ui_counter_word::reused] = 8u; now[ui_counter_word::refined] = 9u;
    now[ui_counter_word::decided + 12] = 10u;
    ui_counters gpu;
    gpu.add_gpu_delta(now, before);
    require(gpu[n::detection_frames] == 3 && gpu.decided(4) == 2 && gpu.decided(10) == 6 &&
        gpu[n::none + ui_no_mask::ambiguous] == 5 && gpu[n::none + ui_no_mask::unaccepted] == 3 &&
        gpu[n::untrusted_inferred] == 2 && gpu[n::full_alpha] == 4 && gpu[n::contradicted] == 7 && gpu[n::reused] == 8 &&
        gpu[n::refined] == 9 && gpu.decided(12) == 10 && !gpu.decided(0), "GPU counter deltas are wrong");
    // A session sums every commit and keeps the latest tick.
    alpha_auto_policy session;
    ui_counters first, second;
    first[n::auto_frames] = first[n::detection_frames] = 3; first.through_ms = 500;
    second[n::auto_frames] = second[n::held_generated] = 2; second.through_ms = 400;
    session.add_counters(first);
    session.add_counters(second);
    const auto total = session.counters();
    require(total[n::auto_frames] == 5 && total[n::detection_frames] == 3 && total[n::held_generated] == 2 &&
        total.through_ms == 500 && total.reconciled(), "The session did not sum its committed counters");

    // Earned once, however many samples confirm it; revoked by an exact
    // pair's one-way contradiction.
    alpha_auto_policy earned;
    for (std::uint64_t tick = 10000; tick <= 13000; tick += 1000) feed(earned, sample().alpha(kind::backbuffer, 200), tick);
    require(accepts(earned, kind::backbuffer) && earned.counters()[n::trust_earned] == 1, "Earning was not counted once");
    for (std::uint64_t tick = 14000; tick <= 16000; tick += 1000)
      feed(earned, sample().alpha(kind::backbuffer, pixels).pair(100, 900).one_way(kind::backbuffer, pixels, 800), tick);
    require(!accepts(earned, kind::backbuffer) && earned.counters()[n::trust_revoked_exact] == 1 &&
        !earned.counters()[n::trust_revoked_declared], "A one-way contradiction was not counted as its revocation");
    // Presented alpha disagreeing with an accepted UIAlpha.
    alpha_auto_policy presented;
    for (std::uint64_t tick = 10000; tick <= 12000; tick += 1000)
      feed(presented, sample().alpha(kind::ui_alpha, 100).alpha(kind::backbuffer, 100), tick);
    require(accepts(presented, kind::ui_alpha) && accepts(presented, kind::backbuffer) &&
        presented.counters()[n::trust_earned] == 2, "Two sources were not accepted");
    for (std::uint64_t tick = 13000; tick <= 15000; tick += 1000)
      feed(presented, sample().alpha(kind::ui_alpha, 100).alpha(kind::backbuffer, 300), tick);
    require(accepts(presented, kind::ui_alpha) && !accepts(presented, kind::backbuffer) &&
        presented.counters()[n::trust_revoked_declared] == 1 && !presented.counters()[n::trust_revoked_exact],
      "A declared-coverage disagreement was not counted as its revocation");
    // Restored acceptance lapses unless this session earns it again, and an
    // earn that confirms it counts.
    alpha_auto_policy lapsing;
    lapsing.restore("backbuffer:24:srgb");
    feed(lapsing, sample().alpha(kind::backbuffer, 0), 1000);
    feed(lapsing, sample().alpha(kind::backbuffer, 0), 61000);
    require(!accepts(lapsing, kind::backbuffer) && lapsing.counters()[n::trust_restored] == 1 &&
        lapsing.counters()[n::trust_lapsed] == 1, "A restore or a provisional lapse was not counted");
    alpha_auto_policy confirmed;
    confirmed.restore("backbuffer:24:srgb,current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) feed(confirmed, sample().alpha(kind::backbuffer, 200), tick);
    require(confirmed.counters()[n::trust_restored] == 2 && confirmed.counters()[n::trust_earned] == 1,
      "Confirming a restored source was not counted as earned");
  }

  // The pure pieces the renderer and the sequence replay share
  // (game3d_ui_temporal.h): the decode of a sample's decision texels and the
  // counts it commits.
  void sample_decode_and_commit_are_shared() {
    namespace word = ui_detection::decision_word;
    std::array<std::uint32_t, 4 * ui_detection::still_decision_texels> t{};
    t[word::source] = 10; t[word::covered] = 995; t[word::pixels] = 1000; t[word::matching_tiles] = 7;
    t[word::candidates] = 0x7a; t[word::hudless_changed] = 9; t[word::hudless_unchanged] = 900;
    t[word::hudless_invalid] = 1; t[word::hudless_lit] = 800; t[word::accepted] = 0x42;
    for (std::uint32_t i = 0; i != 4; ++i) { t[word::alpha_covered + i] = 10 + i; t[word::alpha_invalid + i] = 20 + i; }
    t[word::alpha_opaque] = 30; t[word::alpha_opaque + 1] = 31;
    const float d = .5f, pre_ui_d = .3f;
    t[word::scene_n] = 400; std::memcpy(&t[word::scene_d], &d, sizeof(d));
    t[word::scene_state] = 1u | 2u | std::uint32_t(ui_detection::scene_verdict::visible) << 2; t[word::scene_decided] = 600;
    t[word::pre_ui_scene_n] = 300; std::memcpy(&t[word::pre_ui_scene_d], &pre_ui_d, sizeof(pre_ui_d));
    t[word::pre_ui_scene_state] = 3u; t[word::pre_ui_scene_image] = ui_detection::pre_ui_image::layer;
    t[word::opaque_backbuffer] = 997; t[word::opaque_current] = 998; t[word::claims] = 0xc0u;
    t[word::h1] = 10u | ui_detection::h1_applied;
    t[word::layer_covered] = 995; t[word::layer_invalid] = 2; t[word::layer_opaque] = 990; t[word::valid_bits] = 0x4a;
    for (std::uint32_t i = 0; i != 3; ++i) { t[word::strong + i] = 40 + i; t[word::contradicted + i] = 50 + i; }
    t[word::refused] = candidate::backbuffer;
    t[word::frame_reason] = std::uint32_t(ui_no_mask::unaccepted) | ui_detection::frame_reason_reused;
    t[word::pre_ui_match] = 960; t[word::pre_ui_image_lit] = 880; t[word::presented_lit] = 890; t[word::presented_lit_differs] = 7;
    t[word::still_cells] = 35100; t[word::still_compared] = 36864;
    const auto sample = ui_temporal::decode_detection_sample(t.data(), t.size(), 1500, 9, true);
    const auto &e = sample.evidence;
    require(sample.source_kind == 10 && sample.enabled && sample.state == alpha_auto_state::automatic_on &&
        sample.covered == 995 && sample.pixels == 1000 && sample.sample_tick_ms == 1500 && sample.sample_sequence == 9 &&
        sample.accepted_samples == 9 && e.matching_tiles == 7 && e.candidates == 0x7a && e.hudless_changed == 9 &&
        e.hudless_unchanged == 900 && e.hudless_invalid == 1 && e.hudless_lit == 800 && e.accepted == 0x42 &&
        e.alpha_covered[3] == 13 && e.alpha_invalid[0] == 20 && e.alpha_opaque[1] == 31 && e.layer_covered == 995 &&
        e.layer_invalid == 2 && e.layer_opaque == 990 && e.valid_bits == 0x4a,
      "The decision texels did not decode");
    require(e.strong == std::array<std::uint32_t, 3>{40, 41, 42} && e.contradicted == std::array<std::uint32_t, 3>{50, 51, 52} &&
        e.refused == candidate::backbuffer && e.frame_reason == ui_no_mask::unaccepted && e.reused,
      "The one-way counts, refused candidate or frame reason did not decode");
    require(e.inferred_opaque == std::array<std::uint32_t, 2>{997, 998} && e.claims == 0xc0u && e.s1_source == 10 &&
        e.h1_applied, "Texel 10 did not decode");
    require(e.pre_ui_match == 960 && e.pre_ui_image_lit == 880 && e.presented_lit == 890 && e.presented_lit_differs == 7,
      "Texel 11 did not decode");
    require(e.still_cells == 35100 && e.still_compared == 36864, "Texel 12 (H2) did not decode");
    const auto revision4 = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::pre_ui_decision_texels, 1500, 9, true);
    require(revision4.evidence.pre_ui_match == 960 && !revision4.evidence.still_cells && !revision4.evidence.still_compared,
      "Texel 12 decoded from fewer than 52 words");
    // The scene guard reads the same texel 12 and the reused bit (H2).
    const auto guarded = scene_guard::sample_of(t.data(), t.size(), 1500, true);
    require(guarded.still_cells == 35100 && guarded.still_compared == 36864 && guarded.reused,
      "The scene guard did not decode texel 12 or the reused bit");
    const auto revision3 = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::h1_decision_texels, 1500, 9, true);
    require(revision3.evidence.h1_applied && !revision3.evidence.pre_ui_match && !revision3.evidence.presented_lit_differs,
      "Texel 11 decoded from fewer than 48 words");
    const auto revision2 = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::judgment_decision_texels, 1500, 9, true);
    require(revision2.evidence.reused && !revision2.evidence.claims && !revision2.evidence.h1_applied &&
        !revision2.evidence.inferred_opaque[0], "Texel 10 decoded from fewer than 44 words");
    const auto layer_only = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::layer_decision_texels, 1500, 9, true);
    require(layer_only.evidence.valid_bits == 0x4a && !layer_only.evidence.strong[0] && !layer_only.evidence.refused &&
        layer_only.evidence.frame_reason == ui_detection::frame_reason_decided && !layer_only.evidence.reused,
      "Texels 8 and 9 decoded from fewer than 40 words");
    require(e.scene.n == 400 && e.scene.d == .5f && e.scene.valid && e.scene.ran &&
        e.scene.verdict == ui_detection::scene_verdict::visible && e.scene.decided == 600 && e.pre_ui_scene.n == 300 &&
        e.pre_ui_scene.d == .3f && e.pre_ui_scene.valid && e.pre_ui_scene.ran &&
        e.pre_ui_scene.verdict == ui_detection::scene_verdict::visible && e.pre_ui_image == ui_detection::pre_ui_image::layer,
      "The scene texels did not decode");
    const auto unsupported = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::scene_decision_texels, 1500, 9, false);
    require(!unsupported.evidence.scene.ran && !unsupported.evidence.scene.n && !unsupported.evidence.layer_covered &&
        !unsupported.evidence.valid_bits && unsupported.evidence.accepted == 0x42,
      "Scene texels decoded without scene evidence, or layer texels without texel 7");
    require(!ui_temporal::decode_detection_sample(t.data(), 4, 1500, 9, false).pixels,
      "Too few texels decoded");
    // 995 of 1000 pixels from the layer (source 10) is a whole-frame alpha; its
    // measured verdict is counted with the accepted whole-frame decisions,
    // apart from H1's (8).
    namespace n = ui_counter;
    require(ui_temporal::full_alpha(sample) && ui_temporal::whole_frame(sample), "A 99.5% layer decision was not whole-frame");
    std::array<std::uint32_t, ui_counter_word::count> before{}, after{};
    after[ui_counter_word::detection_frames] = 3; after[ui_counter_word::full_alpha] = 2;
    ui_counters cpu_then, cpu_now;
    cpu_now[n::auto_frames] = 4; cpu_now[n::held_generated] = 1;
    auto delta = ui_temporal::sample_counters(sample, cpu_now, cpu_then, after, before, 1500);
    require(delta[n::auto_frames] == 4 && delta[n::detection_frames] == 3 && delta[n::full_alpha] == 2 &&
        delta[n::full_alpha_d_visible] == 1 && !delta[n::full_d_visible] && delta[n::samples] == 1 &&
        delta.through_ms == 1500 && delta.reconciled(), "A whole-frame alpha sample did not commit its counts");
    // full_d counts H1 samples (8) only, by the verdict they measured, and
    // the scene group the guard's observation of the sample.
    auto h1 = sample;
    h1.source_kind = 8;
    scene_guard::observation observed;
    observed.released = true;
    observed.refuted = 2;
    delta = ui_temporal::sample_counters(h1, cpu_now, cpu_then, after, before, 1500, observed);
    require(delta[n::full_d_visible] == 1 && !delta[n::full_alpha_d_visible] && !ui_temporal::whole_frame(h1) &&
        !delta[n::scene_entered] && delta[n::scene_released] == 1 && delta[n::scene_refuted] == 2,
      "An H1 sample was counted as a whole-frame decision, or its observation was not counted");
    // H2: the guard's observation of the sample adds to the renderer's own
    // ends of a run (a scope or identity change) in its CPU counts.
    {
      auto cpu_still = cpu_now;
      cpu_still[n::still_released] = 1;
      cpu_still[n::still_short] = 2;
      scene_guard::observation still_observed;
      still_observed.still.entered = still_observed.still.short_run = true;
      const auto still_delta = ui_temporal::sample_counters(sample, cpu_still, cpu_then, after, before, 1500, still_observed);
      require(still_delta[n::still_entered] == 1 && still_delta[n::still_released] == 1 && still_delta[n::still_short] == 3 &&
          still_delta.reconciled(), "H2's observation and the renderer's own ends were not both counted");
    }
    // The exact full change set (6) is an accepted whole-frame decision.
    auto full_set = sample;
    full_set.source_kind = 6;
    full_set.covered = pixels;
    full_set.evidence.scene.verdict = ui_detection::scene_verdict::hidden;
    observed = {};
    observed.entered = true;
    delta = ui_temporal::sample_counters(full_set, cpu_now, cpu_then, after, before, 1500, observed);
    require(ui_temporal::whole_frame(full_set) && !ui_temporal::full_alpha(full_set) && delta[n::full_alpha_d_hidden] == 1 &&
        !delta[n::full_d_hidden] && delta[n::scene_entered] == 1, "A source-6 sample was not counted as whole-frame");
    for (const std::uint32_t retired : {7u, 9u}) {
      auto old = sample;
      old.source_kind = retired;
      delta = ui_temporal::sample_counters(old, cpu_now, cpu_then, after, before, 1500);
      require(!delta[n::full_d_visible] && !delta[n::full_alpha_d_visible], "A retired source was counted as full-frame");
    }
    auto partial = sample;
    partial.covered = 980;
    partial.evidence.scene.valid = false;
    delta = ui_temporal::sample_counters(partial, cpu_now, cpu_then, after, before, 1500);
    require(!ui_temporal::full_alpha(partial) && !ui_temporal::whole_frame(partial) && !delta[n::full_alpha_d_invalid] &&
        !delta[n::full_d_invalid], "A 98% alpha sample was counted as whole-frame");
    // The V1-invalid alpha bits a hold reads from a sample.
    alpha_auto_decision::detection_evidence invalid;
    invalid.candidates = 0x4fu; invalid.alpha_invalid = {11, 10, 0, 50}; invalid.layer_invalid = 11;
    require(invalid_alpha_bits(invalid, 1000) == (0x1u | 0x8u | 0x40u), "V1-invalid alpha bits were wrong");
  }
} // namespace

int main() {
  try {
    mode_is_auto_until_a_manual_edit_and_auto_again_after_it();
    declared_sources_are_accepted_by_one_selective_sample();
    inferred_sources_need_a_steady_run();
    an_invalid_declared_alpha_voids_earning();
    acceptance_is_per_signature();
    restore_is_provisional_and_legacy_entries_are_discarded();
    forget_clears_the_ledger();
    revocation_is_per_signature();
    exact_change_sets_earn();
    manual_on_is_a_session_override();
    status_follows_the_winner_and_scope();
    samples_that_flatten_learn_nothing();
    temporal_state_holds();
    scene_guard_holds_refutes_and_scopes();
    pre_ui_proof_is_earned_by_pixels();
    ui_counters_are_formatted_and_trust_events_counted();
    sample_decode_and_commit_are_shared();
    std::puts("Source alpha session: 17 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Source alpha session failed: %s\n", error.what());
    return 1;
  }
}
