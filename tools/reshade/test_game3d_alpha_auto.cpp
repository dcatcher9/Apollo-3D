// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_alpha_auto.h"
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
    feed(late, sample().alpha(kind::current, 1000), 30000);
    feed(late, sample().alpha(kind::current, 1000), 89999);
    require(accepts(late, kind::current), "The lapse clock started before the source was offered valid");
    feed(late, sample().alpha(kind::current, 1000), 90000);
    require(!accepts(late, kind::current), "The lapse clock did not start at the first valid offer");
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
    feed(rejected, sample().alpha(kind::ui_color, 1000), 1000);
    for (std::uint64_t tick = 2000; tick <= 100000; tick += 1000)
      feed(rejected, sample().alpha(kind::ui_color, 0, 600).alpha(kind::current, 400), tick);
    require(accepts(rejected, kind::ui_color) && !rejected.counters()[ui_counter::trust_lapsed],
      "A restored tag lapsed while it was offered but invalid");
    feed(rejected, sample().alpha(kind::ui_color, 1000), 100500);
    feed(rejected, sample().alpha(kind::ui_color, 1000), 159499);
    require(accepts(rejected, kind::ui_color), "The paused clock did not resume at the next valid offer");
    feed(rejected, sample().alpha(kind::ui_color, 1000), 159500);
    require(!accepts(rejected, kind::ui_color) && rejected.counters()[ui_counter::trust_lapsed] == 1,
      "The paused clock never lapsed");
    // Invalid offers never extend the clock: a tag invalid every other
    // second lapses after 60 s of valid offers in total.
    alpha_auto_policy alternating;
    alternating.restore("ui_color:87:srgb");
    for (std::uint64_t tick = 1000; tick <= 120000; tick += 1000)
      feed(alternating, (tick / 1000) % 2 ? sample().alpha(kind::ui_color, 1000) : sample().alpha(kind::ui_color, 0, 600), tick);
    require(accepts(alternating, kind::ui_color), "An alternating tag lapsed before 60 s of valid offers");
    feed(alternating, sample().alpha(kind::ui_color, 1000), 121000);
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
    feed(inferred, sample().alpha(kind::current, 1000), 1000);
    for (std::uint64_t tick = 2000; tick <= 60000; tick += 1000) feed(inferred, sample().alpha(kind::current, 0, 600), tick);
    feed(inferred, sample().alpha(kind::current, 1000), 61000);
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
    require(policy.accepted(0x7fu, signatures()) == ui_selection::candidate_bits, "Manual On did not accept the offered candidates");
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
    const auto layer = ui_detection::layer_detection_flags(false);
    temporal::detection_state state;
    require(!state.status_key(), "A state without adopted inputs had a winner");
    // Resident Evil Requiem with FG: an accepted UI color tag (2) and current
    // alpha (8); its HUD-less pair (16) joins on some Presents only.
    state.adopt(2u, 0u, 2u | 8u);
    const auto tag = state.status_key();
    state.adopt(2u | 8u | 16u, 0u, 2u | 8u);
    require(tag == candidate::ui_color && state.status_key() == tag, "A pairing or inferred alpha beside the tag moved the winner");
    state.adopt(2u | 48u, 0u, 2u | 8u | 16u);
    require(state.status_key() == candidate::ui_color, "An accepted HUD-less pair beside the tag won");
    // The first accepted candidate in draw order wins, whatever is offered
    // beside it; unaccepted candidates and stored flags never key it.
    state.adopt(0x40u | 4u | 48u, layer, 0x40u | 4u);
    require(state.status_key() == candidate::layer, "The accepted layer did not win before Backbuffer");
    state.adopt(0x40u | 4u, layer | ui_detection::stored_hdr_headroom, 4u);
    require(state.status_key() == candidate::backbuffer, "An unaccepted layer won");
    state.adopt(16u | 32u, 0u, 16u);
    require(state.status_key() == candidate::hudless, "An accepted HUD-less pair did not win alone");
    state.adopt(2u | 4u | 16u, 0u, 0u);
    require(!state.status_key(), "Nothing accepted had a winner");
    // Adoption keeps the latest sample: status_fresh decides what it describes.
    alpha_auto_source scope;
    scope.epoch = 3; scope.revision = 4; scope.viewport = 5; scope.now_ms = 1200;
    state.adopt(2u, 0u, 2u);
    state.latest.sample_tick_ms = 1000;
    state.latest_source = scope;
    state.latest_key = candidate::ui_color;
    state.adopt(2u | 8u | 16u, 0u, 2u | 8u);
    require(state.latest.sample_tick_ms == 1000 && state.status_fresh(scope), "A pairing beside the winner dropped or staled the sample");
    state.adopt(8u, 0u, 8u);
    require(state.latest.sample_tick_ms == 1000 && !state.status_fresh(scope), "A sample described another winner");
    state.adopt(2u, 0u, 2u);
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

  // Hidden-scene gates (docs/reshade-sbs.md, hidden-scene evidence) open only
  // when no source 1-6 or 10 decided, for an unaccepted UIAlpha or offscreen
  // UI layer nearly opaque without an invalid pixel, or a HUD-less image
  // differing on at least 90% of the frame.
  void hidden_scene_gates_and_samples_that_flatten_learn_nothing() {
    namespace detection = ui_detection;
    const auto gates = [&](std::uint32_t source, std::uint32_t offered, std::uint32_t accepted,
        std::array<std::uint32_t, 2> opaque, std::uint32_t changed, std::array<std::uint32_t, 2> invalid = {}) {
      return detection::scene_gates_of(source, offered, accepted, pixels, invalid, opaque, changed);
    };
    require(gates(0, 0x40, 0, {0, 990}, 0).layer && gates(8, 0x40, 0, {0, 1000}, 0).layer,
      "An unaccepted opaque layer did not open the layer route");
    require(!gates(0, 0x40, 0, {0, 989}, 0).layer, "A layer under 99% opaque opened the layer route");
    require(!gates(0, 2, 0, {1000, 1000}, 0).layer, "A tagged UI color opened the layer route");
    require(!gates(0, 0x40, 0x40, {0, 1000}, 0).layer, "An accepted layer opened the layer route");
    require(!gates(0, 0x40, 0, {0, 1000}, 0, {0, 1}).layer, "A layer with an invalid pixel opened the layer route");
    require(gates(0, 1, 0, {1000, 0}, 0).layer && !gates(0, 1, 1, {1000, 0}, 0).layer,
      "UIAlpha did not open the layer route only while unaccepted");
    require(!gates(0, 4 | 8, 0, {1000, 1000}, 0).layer, "Presented alpha opened the layer route");
    for (const std::uint32_t decided : {1u, 2u, 3u, 4u, 5u, 6u, 10u})
      require(!gates(decided, 0x40 | 16, 0, {0, 1000}, 1000).layer && !gates(decided, 16, 0, {}, 1000).hudless,
        "A decided frame opened a hidden-scene route");
    require(gates(0, 16, 0, {}, 900).hudless && gates(9, 48, 0, {}, 1000).hudless && !gates(0, 16, 0, {}, 899).hudless &&
        !gates(0, 0, 0, {}, 1000).hudless, "The HUD-less route did not open at 90% changed pixels exactly");
    require(detection::scene_visible(.25f) && !detection::scene_visible(.2499f), "The visible bound moved");
    // Per input: those that opened the layer route, and the V1-valid inputs
    // offered below that opacity (overlays), whatever source decided. An
    // invalid layer (Stellar Blade's SDR scene image) is no overlay.
    require(gates(0, 0x41, 0, {1000, 989}, 0).layer_slots == 1u && gates(0, 0x41, 0, {1000, 989}, 0).overlay_slots == 2u,
      "The layer route's inputs were not told apart");
    require(gates(10, 0x40, 0x40, {0, 100}, 0).overlay_slots == 2u && !gates(10, 0x40, 0x40, {0, 100}, 0).layer_slots,
      "A layer that decided its own selective mask was not shown an overlay");
    require(!gates(0, 2, 0, {100, 100}, 0).overlay_slots && !gates(0, 4 | 8, 0, {0, 0}, 0).overlay_slots,
      "A tagged UI color or presented alpha counted as a layer-route input");
    require(!gates(0, 0x40 | 4, 0, {0, 0}, 0, {0, 600}).overlay_slots && gates(0, 0x40 | 4, 0, {0, 0}, 0, {0, 10}).overlay_slots == 2u,
      "An invalid layer counted as an overlay, or a valid one did not");
    // Holds and refutations follow the routes' inputs, not a HUD-less pairing
    // or a UI color tag.
    const auto layer = detection::layer_detection_flags(false);
    require(detection::scene_route_key(0x40, layer, 0) == detection::scene_route_key(0x40 | 16, layer, 0) &&
        detection::scene_route_key(0x40 | 16, layer, 0) == detection::scene_route_key(0x40 | 48, layer, 0) &&
        detection::scene_route_key(0x40 | 2 | 4 | 8, layer, 2 | 4 | 8) == detection::scene_route_key(0x40, layer, 0),
      "A HUD-less pairing, a tag or presented alpha changed the routes' inputs");
    require(detection::scene_route_key(0x40, layer, 0) != detection::scene_route_key(0x40, layer, 0x40) &&
        detection::scene_route_key(0x40, layer, 0) != detection::scene_route_key(0x40, 0u, 0) &&
        detection::scene_route_key(0x40, layer, 0) != detection::scene_route_key(0, 0u, 0) &&
        detection::scene_route_key(1, 0u, 0) != detection::scene_route_key(1, 0u, 1),
      "Acceptance, an input or its stored flags did not change the routes' inputs");

    // The counts of a sample that decided 8 or 9 cover the whole frame with an
    // unaccepted candidate: they are never selective, so they never earn.
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
  // real Present always detects and pushes its per-frame bits. The CPU-held
  // hidden-scene verdicts. The runtime tests prove the renderer calls it;
  // these pin its rules without a GPU.
  void temporal_state_holds_and_scene_verdicts() {
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
    state.adopt(16u | 32u, 0u, 16u);
    state.detected(scope, present_identity{false, 5});
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
    state.detected(scope, present_identity{false, 6});
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
    accepted.adopt(4u, 0u, 4u);
    accepted.detected(scope, present_identity{});
    const auto missing = accepted.arbitrate(present_identity{}, scope, 0u);
    require(missing.detect && !missing.adopt && missing.per_frame == ui_detection::per_frame_accepted_missing,
      "A missing accepted candidate was not flagged, or the frame adopted");
    require(accepted.arbitrate(present_identity{}, scope, 8u).per_frame == ui_detection::per_frame_accepted_missing,
      "An unaccepted current alpha stood in for the missing Backbuffer");
    require(!accepted.arbitrate(present_identity{}, scope, 4u | 8u).per_frame && accepted.arbitrate(present_identity{}, scope, 4u).adopt,
      "An offered accepted candidate was flagged missing");
    temporal::detection_state unaccepted;
    unaccepted.adopt(4u | 0x40u, ui_detection::layer_detection_flags(false), 0x40u);
    unaccepted.detected(scope, present_identity{});
    require(!unaccepted.arbitrate(present_identity{}, scope, 0x40u).per_frame &&
        unaccepted.arbitrate(present_identity{}, scope, 4u).per_frame == ui_detection::per_frame_accepted_missing,
      "Missing candidates were flagged by the wrong acceptance");
    // A generated Present never flags or adopts, whatever it offers.
    const auto generated = accepted.arbitrate(present_identity{true, 0}, scope, 1u);
    require(generated.hold && !generated.per_frame && !generated.adopt, "A generated Present detected or adopted");

    // One valid hidden sample holds the layer route for hold_ms from its tick;
    // a visible one releases it and refutes the input until it is an overlay.
    const auto layer = ui_detection::layer_detection_flags(false);
    temporal::detection_state scene;
    scene.adopt(0x40u, layer, 0u);
    alpha_auto_decision sample;
    sample.pixels = 1000; sample.sample_tick_ms = 2000;
    sample.evidence.candidates = 0x40u; sample.evidence.layer_opaque = 1000;
    sample.evidence.scene.valid = true; sample.evidence.scene.verdict = ui_detection::scene_verdict::hidden;
    scene.observe_scene(sample, true);
    require(scene.scene_gate_open && scene.scene_holds(2500) == ui_detection::per_frame_scene_hold && !scene.scene_holds(2501),
      "One hidden sample did not hold the layer route for hold_ms");
    auto shadow = sample;
    shadow.sample_tick_ms = 2100; shadow.evidence.scene.verdict = ui_detection::scene_verdict::visible;
    scene.observe_scene(shadow, false);
    require(scene.scene_holds(2500), "Shadow-only evidence changed a route");
    sample.sample_tick_ms = 2200; sample.evidence.scene.verdict = ui_detection::scene_verdict::visible;
    scene.observe_scene(sample, true);
    require(!scene.scene_holds(2300) && scene.scene_refuted_slots == 2u, "A visible sample did not release and refute");
    sample.sample_tick_ms = 2300; sample.evidence.scene.verdict = ui_detection::scene_verdict::hidden;
    scene.observe_scene(sample, true);
    require(!scene.scene_holds(2400), "A refuted input held the layer route");
    sample.evidence.layer_opaque = 500;
    scene.observe_scene(sample, true);
    require(!scene.scene_refuted_slots, "An input offered as an overlay stayed refuted");
    sample.evidence.layer_opaque = 1000; sample.sample_tick_ms = 2400;
    scene.observe_scene(sample, true);
    require(scene.scene_holds(2900), "An input shown an overlay could not hold the route again");
    sample.sample_tick_ms = 2600; sample.evidence.scene.valid = false;
    scene.observe_scene(sample, true);
    require(scene.scene_holds(2900) && !scene.scene_holds(2901), "Invalid evidence renewed or released a hold");
    // A HUD-less pairing or acceptance beside the route keeps its hold; a
    // new route drops it (scene_route_key, S2b).
    scene.adopt(0x40u | 16u, layer, 0u);
    require(scene.scene_holds(2900), "A HUD-less pairing dropped the route's hold");
    scene.adopt(0x40u, layer, 0x40u);
    require(!scene.scene_holds(2900), "A new route kept its hold");
    // An accepted layer opens no route.
    temporal::detection_state accepted_layer;
    auto decided = sample;
    decided.evidence.accepted = 0x40u; decided.evidence.scene.valid = true;
    accepted_layer.observe_scene(decided, true);
    require(!accepted_layer.scene_gate_open && !accepted_layer.scene_holds(2600), "An accepted layer opened the layer route");
    // A scope change and an inactive render clear the holds; an inactive
    // render also ends the T1 chain.
    scene.scene_hold_until = {3000, 3000};
    alpha_auto_source moved;
    moved.epoch = 1;
    scene.enter_scope(moved);
    require(!scene.scene_holds(2500), "A new scope kept a hold");
    scene.scene_hold_until = {3000, 3000}; scene.scene_refuted_slots = 2u;
    scene.detected(scope, present_identity{false, 9});
    scene.inactive();
    require(!scene.scene_holds(2500) && !scene.scene_refuted_slots && !scene.have_decision &&
        scene.arbitrate(present_identity{true, 9}, scope, 0u).kind == hold_kind::unavailable &&
        scene.arbitrate(present_identity{false, 10}, scope, 0u).per_frame == ui_detection::per_frame_hold_reset,
      "An inactive render kept a hold, refutation or decision");
  }

  // Exact UI counters (game3d_ui_counters.h): the log text, the accounting
  // identity, wrap-safe GPU deltas, session totals and acceptance events.
  void ui_counters_are_formatted_and_trust_events_counted() {
    namespace n = ui_counter;
    ui_counters c;
    c[n::auto_frames] = 10; c[n::detection_frames] = 6;
    c[n::held_generated] = 2; c[n::held_none] = 1; c[n::reused] = 1;
    c[n::inactive_no_candidates] = 1;
    c[n::decided + 0] = 2; c[n::decided + 5] = 3; c[n::decided + 6] = 1; c[n::decided + 10] = 4;
    c[n::none + ui_no_mask::difference_failed] = 1; c[n::none + ui_no_mask::gate_no_hold] = 1;
    c[n::none + ui_no_mask::unaccepted] = 2;
    c[n::depth_not_current] = 1; c[n::full_d_hidden] = 1; c[n::inexact_difference] = 3; c[n::trust_earned] = 1;
    c[n::trust_discarded] = 2; c[n::trust_revoked_exact] = 1; c[n::trust_forgotten] = 3; c[n::contradicted] = 2;
    c[n::full_alpha] = 2; c[n::full_alpha_d_visible] = 1;
    c[n::samples] = 4; c.through_ms = 12345;
    require(c.reconciled() && c.held() == 3 && c.inactive() == 1, "The counters' accounting identity is wrong");
    // reused frames are detection frames, not holds.
    ++c[n::reused];
    require(c.reconciled(), "A reused frame changed the accounting identity");
    --c[n::reused];
    require(format_ui_counters(c) ==
        "auto_frames=10 detection_frames=6 held={generated=2 none=1} reused=1 "
        "inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 9=0 10=4} "
        "none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 gate_no_hold=1 "
        "no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 9=0 depth_not_current=1} full_d={hidden=1 ambiguous=0 visible=0 invalid=0} "
        "untrusted_inferred=0 inexact_difference=3 contradicted=2 presented_over_dedicated=0 full_alpha=2 "
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
    now[ui_counter_word::contradicted] = 7u; now[ui_counter_word::reused] = 8u;
    ui_counters gpu;
    gpu.add_gpu_delta(now, before);
    require(gpu[n::detection_frames] == 3 && gpu.decided(4) == 2 && gpu.decided(10) == 6 &&
        gpu[n::none + ui_no_mask::ambiguous] == 5 && gpu[n::none + ui_no_mask::unaccepted] == 3 &&
        gpu[n::untrusted_inferred] == 2 && gpu[n::full_alpha] == 4 && gpu[n::contradicted] == 7 && gpu[n::reused] == 8 &&
        !gpu.decided(0), "GPU counter deltas are wrong");
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
    feed(lapsing, sample().alpha(kind::backbuffer, pixels), 1000);
    feed(lapsing, sample().alpha(kind::backbuffer, pixels), 61000);
    require(!accepts(lapsing, kind::backbuffer) && lapsing.counters()[n::trust_restored] == 1 &&
        lapsing.counters()[n::trust_lapsed] == 1, "A restore or a provisional lapse was not counted");
    alpha_auto_policy confirmed;
    confirmed.restore("backbuffer:24:srgb,current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) feed(confirmed, sample().alpha(kind::backbuffer, 200), tick);
    require(confirmed.counters()[n::trust_restored] == 2 && confirmed.counters()[n::trust_earned] == 1,
      "Confirming a restored source was not counted as earned");
  }

  // The pure pieces the renderer and the sequence replay share
  // (game3d_ui_temporal.h): the decode of a sample's decision texels, the
  // counts it commits and when a sample frame measures the scene.
  void sample_decode_and_commit_are_shared() {
    namespace word = ui_detection::decision_word;
    std::array<std::uint32_t, 4 * ui_detection::judgment_decision_texels> t{};
    t[word::source] = 10; t[word::covered] = 995; t[word::pixels] = 1000; t[word::matching_tiles] = 7;
    t[word::candidates] = 0x7a; t[word::hudless_changed] = 9; t[word::hudless_unchanged] = 900;
    t[word::hudless_invalid] = 1; t[word::hudless_lit] = 800; t[word::accepted] = 0x42;
    for (std::uint32_t i = 0; i != 4; ++i) { t[word::alpha_covered + i] = 10 + i; t[word::alpha_invalid + i] = 20 + i; }
    t[word::alpha_opaque] = 30; t[word::alpha_opaque + 1] = 31;
    const float d = .5f, hudless_d = .3f;
    t[word::scene_n] = 400; std::memcpy(&t[word::scene_d], &d, sizeof(d));
    t[word::scene_state] = 1u | 2u | std::uint32_t(ui_detection::scene_verdict::visible) << 2; t[word::scene_decided] = 600;
    t[word::hudless_scene_n] = 300; std::memcpy(&t[word::hudless_scene_d], &hudless_d, sizeof(hudless_d));
    t[word::hudless_scene_state] = 3u;
    t[word::layer_covered] = 995; t[word::layer_invalid] = 2; t[word::layer_opaque] = 990; t[word::valid_bits] = 0x4a;
    for (std::uint32_t i = 0; i != 3; ++i) { t[word::strong + i] = 40 + i; t[word::contradicted + i] = 50 + i; }
    t[word::refused] = candidate::backbuffer;
    t[word::frame_reason] = std::uint32_t(ui_no_mask::unaccepted) | ui_detection::frame_reason_reused;
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
    const auto layer_only = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::layer_decision_texels, 1500, 9, true);
    require(layer_only.evidence.valid_bits == 0x4a && !layer_only.evidence.strong[0] && !layer_only.evidence.refused &&
        layer_only.evidence.frame_reason == ui_detection::frame_reason_decided && !layer_only.evidence.reused,
      "Texels 8 and 9 decoded from fewer than 40 words");
    require(e.scene.n == 400 && e.scene.d == .5f && e.scene.valid && e.scene.ran &&
        e.scene.verdict == ui_detection::scene_verdict::visible && e.scene.decided == 600 && e.hudless_scene.n == 300 &&
        e.hudless_scene.d == .3f && e.hudless_scene.valid && e.hudless_scene.ran, "The scene texels did not decode");
    const auto unsupported = ui_temporal::decode_detection_sample(t.data(), 4 * ui_detection::scene_decision_texels, 1500, 9, false);
    require(!unsupported.evidence.scene.ran && !unsupported.evidence.scene.n && !unsupported.evidence.layer_covered &&
        !unsupported.evidence.valid_bits && unsupported.evidence.accepted == 0x42,
      "Scene texels decoded without scene evidence, or layer texels without texel 7");
    require(!ui_temporal::decode_detection_sample(t.data(), 4, 1500, 9, false).pixels,
      "Too few texels decoded");
    // 995 of 1000 pixels from the layer (source 10) is a whole-frame alpha; its
    // measured verdict is counted apart from the full-frame routes'.
    namespace n = ui_counter;
    require(ui_temporal::full_alpha(sample), "A 99.5% layer decision was not whole-frame");
    std::array<std::uint32_t, ui_counter_word::count> before{}, after{};
    after[ui_counter_word::detection_frames] = 3; after[ui_counter_word::full_alpha] = 2;
    ui_counters cpu_then, cpu_now;
    cpu_now[n::auto_frames] = 4; cpu_now[n::held_generated] = 1;
    auto delta = ui_temporal::sample_counters(sample, cpu_now, cpu_then, after, before, 1500);
    require(delta[n::auto_frames] == 4 && delta[n::detection_frames] == 3 && delta[n::full_alpha] == 2 &&
        delta[n::full_alpha_d_visible] == 1 && !delta[n::full_d_visible] && delta[n::samples] == 1 &&
        delta.through_ms == 1500 && delta.reconciled(), "A whole-frame alpha sample did not commit its counts");
    auto route = sample;
    route.source_kind = 8;
    delta = ui_temporal::sample_counters(route, cpu_now, cpu_then, after, before, 1500);
    require(delta[n::full_d_visible] == 1 && !delta[n::full_alpha_d_visible], "A route sample was counted as alpha");
    auto partial = sample;
    partial.covered = 980;
    partial.evidence.scene.valid = false;
    delta = ui_temporal::sample_counters(partial, cpu_now, cpu_then, after, before, 1500);
    require(!ui_temporal::full_alpha(partial) && !delta[n::full_alpha_d_invalid] && !delta[n::full_d_invalid],
      "A 98% alpha sample was counted as whole-frame");
    // After a whole-frame alpha sample the next sample frame measures, but
    // what it measures is not actionable.
    ui_temporal::detection_state state;
    require(!state.measure_scene(1000, false).run && state.measure_scene(1000, true).run &&
        !state.measure_scene(1000, true).actionable, "The shadow alone measured as actionable");
    state.latest = sample;
    const auto measure = state.measure_scene(1000, false);
    require(measure.run && !measure.actionable, "A whole-frame alpha sample did not measure the next one, or held");
    state.latest = partial;
    state.scene_gate_open = true;
    require(state.measure_scene(1000, false).run && state.measure_scene(1000, false).actionable,
      "An open gate did not measure actionably");
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
    hidden_scene_gates_and_samples_that_flatten_learn_nothing();
    temporal_state_holds_and_scene_verdicts();
    ui_counters_are_formatted_and_trust_events_counted();
    sample_decode_and_commit_are_shared();
    std::puts("Source alpha session: 15 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Source alpha session failed: %s\n", error.what());
    return 1;
  }
}
