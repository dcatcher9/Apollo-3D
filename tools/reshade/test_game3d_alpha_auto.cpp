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
  };
  void feed(alpha_auto_policy &policy, const sample &s, std::uint64_t tick, std::uint32_t color_space = 1) {
    policy.observe(s.e, pixels, tick, signatures(color_space));
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
    // Revocation still evaluates void samples.
    for (const std::uint64_t tick : {10000u, 11000u, 12000u})
      feed(policy, sample().alpha(kind::backbuffer, 1000).alpha(kind::ui_color, 0, 600).pair(10, 900), tick);
    require(!accepts(policy, kind::backbuffer), "A full claim over a visible scene in void samples was not revoked");
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

  // A3 (as before S2a) and the legacy discard: restored entries decide at once
  // but lapse unless earned again within alpha_trust_reconfirm_ms of first
  // being offered valid; anything that is not a key is discarded and counted.
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
  }

  // A2 as before S2a, per signature: a full claim while an exact pair shows
  // at least 75% of the scene, and presented alpha disagreeing by at least
  // 10% of the frame with every accepted, valid UIAlpha, tag or layer.
  void revocation_is_per_signature() {
    alpha_auto_policy policy;
    std::vector<std::string> heard;
    policy.on_change([&](const std::string &stored) { heard.push_back(stored); });
    for (const std::uint64_t tick : {1000u, 2000u, 3000u}) feed(policy, sample().alpha(kind::backbuffer, 200), tick);
    feed(policy, sample().alpha(kind::backbuffer, 200), 3000, 3);
    require(accepts(policy, kind::backbuffer) && heard == std::vector<std::string>{"backbuffer:24:srgb"},
      "Earning was not reported exactly once");
    // A full claim over a menu, or without an exact pair, is no contradiction.
    for (std::uint64_t tick = 4000; tick <= 8000; tick += 1000) {
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(900, 100), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(100, 900, false), tick);
      feed(policy, sample().alpha(kind::backbuffer, 1000), tick);
    }
    require(accepts(policy, kind::backbuffer), "A menu or an inexact pair revoked acceptance");
    // The same evidence interval revokes; a selective sample restarts the doubt.
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 10000);
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 11000);
    feed(policy, sample().alpha(kind::backbuffer, 200), 11500);
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 12000);
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 13000);
    require(accepts(policy, kind::backbuffer), "Doubt survived a selective sample");
    // The same claim in another color space doubts only that signature.
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 13500, 3);
    feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), 14000);
    require(!accepts(policy, kind::backbuffer) && policy.counters()[ui_counter::trust_revoked_full] == 1 &&
        heard.back().empty(), "A full claim over a visible scene was not revoked and reported");
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
        presented.counters()[ui_counter::trust_revoked_presented] == 1,
      "A dedicated UI mask did not revoke presented alpha that covers the scene");
    for (std::uint64_t tick = 8000; tick <= 13500; tick += 500)
      feed(presented, sample().alpha(kind::ui_alpha, tick < 10500 ? 2u : 1000u).alpha(kind::current, tick < 10500 ? 300u : 4u), tick);
    require(!accepts(presented, kind::current), "Contradicted presented alpha earned acceptance again");
    for (const std::uint64_t tick : {14000u, 15000u, 16000u})
      feed(presented, sample().alpha(kind::ui_alpha, 20).alpha(kind::current, 30), tick);
    require(accepts(presented, kind::current), "Presented alpha agreeing with the UI mask could not earn acceptance");
    // An accepted layer is such a mask; an unaccepted or invalid one is none.
    alpha_auto_policy layer;
    layer.restore("ui_layer:10:srgb,backbuffer:24:srgb");
    for (const std::uint64_t tick : {1000u, 2000u, 3000u})
      feed(layer, sample().alpha(kind::ui_layer, 20, 600).alpha(kind::backbuffer, 400), tick);
    require(accepts(layer, kind::backbuffer), "An invalid layer revoked presented alpha");
    for (const std::uint64_t tick : {4000u, 5000u, 6000u})
      feed(layer, sample().alpha(kind::ui_layer, 20).alpha(kind::backbuffer, 400), tick);
    require(!accepts(layer, kind::backbuffer) && accepts(layer, kind::ui_layer),
      "An accepted layer did not revoke disagreeing presented alpha");
    alpha_auto_policy alone;
    alone.restore("current:24:srgb");
    for (std::uint64_t tick = 1000; tick <= 8000; tick += 1000)
      feed(alone, sample().alpha(kind::ui_layer, tick % 2000 ? 2u : 40u).alpha(kind::current, 400), tick);
    require(accepts(alone, kind::current) && !accepts(alone, kind::ui_layer),
      "An unaccepted UI layer revoked presented alpha");
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
      feed(policy, sample().alpha(kind::backbuffer, 1000).pair(250, 750), tick);
    }
    policy.set_manual(false);
    require(!policy.accepted(0x7fu, signatures()), "Manual Off accepted a candidate");
    policy.set_automatic();
    require(policy.stored() == "backbuffer:24:srgb" && heard.empty() && !policy.counters()[ui_counter::trust_earned] &&
        !policy.counters()[ui_counter::trust_revoked_full] && !policy.counters()[ui_counter::trust_lapsed],
      "Manual On earned, revoked, lapsed or persisted");
    require(policy.accepted(0x7fu, signatures()) == candidate::backbuffer, "Auto lost the ledger after manual On");
  }

  void only_the_deciding_inputs_key_a_status_sample() {
    // Resident Evil Requiem with FG: an accepted UI color tag (2) and current
    // alpha (8); its HUD-less pair (16) joins on some Presents only.
    const std::uint32_t accepted = candidate::ui_color | candidate::current;
    require(detection_decision_key(2u, 0u, accepted) == detection_decision_key(2u | 16u, 0u, accepted) &&
        detection_decision_key(2u | 16u, 0u, accepted) == detection_decision_key(2u | 48u, 0u, accepted),
      "A HUD-less pair beside an accepted candidate changed the decision");
    // The accepted declared tag blocks inferred alpha, so a presented channel
    // offered on some Presents only (Stellar Blade in SDR with FG tags the
    // Backbuffer with real Presents) does not change the key.
    require(detection_decision_key(2u | 8u, 0u, accepted) == detection_decision_key(2u, 0u, accepted) &&
        detection_decision_key(2u | 4u, 0u, 2u | 4u) == detection_decision_key(2u, 0u, 2u | 4u),
      "Inferred alpha beside an accepted declared alpha changed the decision");
    require(detection_decision_key(8u, 0u, accepted) != detection_decision_key(2u, 0u, accepted),
      "A different deciding candidate kept the decision");
    // The layer's stored flags key its decision while it is offered, and only then.
    const auto layer = ui_detection::layer_detection_flags(false);
    require(detection_decision_key(0x40u, layer, 0u) != detection_decision_key(0x40u, 0u, 0u) &&
        detection_decision_key(0x40u, layer, 0x40u) != detection_decision_key(0x40u, 0u, 0x40u) &&
        detection_decision_key(0x40u, layer | ui_detection::stored_hdr_headroom, 0x40u) !=
          detection_decision_key(0x40u, layer, 0x40u), "The layer's stored flags did not key its decision");
    require(detection_decision_key(2u, layer, 2u) == detection_decision_key(2u, 0u, 2u) &&
        detection_decision_key(2u, layer, 0u) == detection_decision_key(2u, 0u, 0u),
      "Stored flags keyed a decision without the layer");
    // The tag and the layer have their own bits: one never keys as the other.
    require(detection_decision_key(0x40u, layer, 0x40u) != detection_decision_key(2u, layer, 2u),
      "The UI layer and a tagged UI color shared a decision key");
    // An accepted layer keys alone: neither unaccepted candidates and pairings
    // nor an accepted Backbuffer that frame generation tags on real Presents
    // only change its key.
    require(detection_decision_key(0x40u | 4u, layer, 0x40u) == detection_decision_key(0x40u | 4u | 48u, layer, 0x40u) &&
        detection_decision_key(0x40u | 2u, layer, 0x40u) == detection_decision_key(0x40u, layer, 0x40u) &&
        detection_decision_key(0x40u | 4u, layer, 0x44u) == detection_decision_key(0x40u, layer, 0x44u),
      "A candidate beside an accepted layer changed its key");
    // An accepted HUD-less pair beside an accepted alpha that decides first in
    // draw order: the pair, joining on some Presents only, keys nothing
    // (Resident Evil Requiem and The Witcher 3 with FG after an exact pair
    // was accepted with FG off).
    for (const std::uint32_t alpha : {1u, 2u}) {
      const std::uint32_t both = alpha | 16u;
      require(detection_decision_key(alpha, 0u, both) == detection_decision_key(alpha | 16u, 0u, both) &&
          detection_decision_key(alpha | 16u, 0u, both) == detection_decision_key(alpha | 48u, 0u, both),
        "An accepted HUD-less pair beside an accepted alpha changed the key");
    }
    require(detection_decision_key(1u | 2u, 0u, 3u) == detection_decision_key(1u, 0u, 3u) &&
        detection_decision_key(1u | 2u, 0u, 3u) != detection_decision_key(2u, 0u, 3u),
      "The first accepted alpha in draw order did not key alone");
    // An accepted HUD-less pair: its exactness decides a full change set.
    require(detection_decision_key(16u, 0u, 16u) != detection_decision_key(48u, 0u, 16u) &&
        detection_decision_key(16u | 8u, 0u, 16u) == detection_decision_key(16u, 0u, 16u),
      "An accepted HUD-less pair lost its exactness, or an unaccepted candidate keyed it");
    // Without an accepted candidate every bit, exactness and acceptance matter.
    require(detection_decision_key(2u, 0u, 0u) != detection_decision_key(2u | 16u, 0u, 0u) &&
        detection_decision_key(16u, 0u, 0u) != detection_decision_key(48u, 0u, 0u) &&
        detection_decision_key(2u | 16u, 0u, 2u) != detection_decision_key(2u | 16u, 0u, 0u),
      "An unaccepted decision ignored a candidate, exactness or acceptance");
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
    // Only a session without restored acceptance runs the first-run shadow;
    // the caller says which, and nothing learned later changes it.
    alpha_auto_policy first;
    require(!first.first_run(), "A policy started as a first run without its owner saying so");
    first.set_first_run(true);
    feed(first, sample().alpha(kind::ui_alpha, 100), 1000);
    require(first.first_run() && accepts(first, kind::ui_alpha), "Earning acceptance ended the first-run shadow");
  }

  // The renderer's temporal state (game3d_ui_temporal.h): which render holds
  // the previous mask and why, which inputs a sample still describes, and the
  // CPU-held hidden-scene verdicts. The runtime tests prove the renderer
  // calls it as before; these pin its rules without a GPU.
  void temporal_state_holds_and_scene_verdicts() {
    namespace temporal = ui_temporal;
    using temporal::hold_kind;
    constexpr std::uint32_t max_held = 3;
    temporal::detection_state state;
    // Nothing decided yet: nothing to hold.
    require(!state.arbitrate(0u, 0u, true, max_held).hold, "A generated Present held a mask no decision made");
    // A HUD-less decision makes the mask holdable; a generated Present then
    // holds it for at most max_held Presents.
    state.adopt(16u | 32u, 0u, 0u);
    state.detected();
    require(state.mask_ready && state.exact, "An exact HUD-less decision was not holdable");
    for (std::uint32_t i = 0; i != max_held; ++i) {
      const auto hold = state.arbitrate(0u, 0u, true, max_held);
      require(hold.hold && hold.kind == hold_kind::generated && !hold.cap_reached, "A generated Present did not hold");
      state.held();
    }
    const auto capped = state.arbitrate(0u, 0u, true, max_held);
    require(!capped.hold && capped.cap_reached && capped.kind == hold_kind::generated, "A hold past the cap was not reported");
    // A fresh decision restarts the count; an inexact pair right after an
    // exact decision holds, and a generated Present names the hold first.
    state.detected();
    require(!state.holds, "A fresh decision kept the hold count");
    const auto inexact = state.arbitrate(16u, 0u, false, max_held);
    require(inexact.hold && inexact.kind == hold_kind::inexact_after_exact, "An inexact pair after an exact one did not hold");
    require(state.arbitrate(16u, 0u, true, max_held).kind == hold_kind::generated, "Hold kinds lost their priority");
    require(!state.arbitrate(48u, 0u, false, max_held).hold, "An exact pair was held");
    // An accepted candidate that decided and is missing holds; offered, it decides.
    temporal::detection_state accepted;
    accepted.adopt(4u, 0u, 4u);
    accepted.detected();
    require(accepted.mask_ready && !accepted.exact, "An accepted Backbuffer decision was not holdable");
    const auto missing = accepted.arbitrate(0u, 4u, false, max_held);
    require(missing.hold && missing.kind == hold_kind::trusted_missing, "A missing accepted candidate was not held");
    require(!accepted.arbitrate(4u, 4u, true, max_held).hold, "An accepted candidate in this frame did not decide by itself");
    // Acceptance is never inherited: an offered, unaccepted tag does not
    // stand in for the missing accepted layer.
    temporal::detection_state layer_state;
    const auto layer = ui_detection::layer_detection_flags(false);
    layer_state.adopt(0x40u, layer, 0x40u);
    layer_state.detected();
    const auto tag_only = layer_state.arbitrate(2u, 0u, false, max_held);
    require(tag_only.hold && tag_only.kind == hold_kind::trusted_missing, "An unaccepted tag stood in for the missing layer");
    require(!layer_state.arbitrate(2u, 2u, false, max_held).hold, "An accepted valid tag did not decide by itself");
    // Unaccepted alpha never makes a holdable mask.
    temporal::detection_state unaccepted;
    unaccepted.adopt(4u, 0u, 0u);
    unaccepted.detected();
    require(!unaccepted.mask_ready && !unaccepted.arbitrate(0u, 0u, true, max_held).hold, "An unaccepted mask was held");
    // An accepted candidate the latest sample read V1-invalid does not stop the
    // hold of the mask another candidate made (V1 generalises the old
    // set-aside layer).
    temporal::detection_state aside;
    aside.adopt(0x40u | 4u, layer, 0x40u | 4u);
    aside.detected();
    aside.latest.pixels = 1000; aside.latest.evidence.candidates = 0x40u | 4u; aside.latest.evidence.layer_invalid = 600;
    require(aside.arbitrate(0x40u, 0x40u | 4u, true, max_held).hold, "An invalid accepted layer stopped the hold");
    aside.latest.evidence.layer_invalid = 10;
    require(!aside.arbitrate(0x40u, 0x40u | 4u, true, max_held).hold, "A valid accepted layer did not decide by itself");
    aside.latest.evidence.candidates = 2u | 4u; aside.latest.evidence.alpha_invalid = {0, 600, 0, 0};
    require(aside.arbitrate(2u, 2u | 4u, true, max_held).hold, "An invalid accepted tag stopped the hold");
    // A change of the deciding inputs drops the sample; a HUD-less pairing
    // beside an accepted candidate does not, and neither changes the route.
    temporal::detection_state keyed;
    keyed.adopt(4u, 0u, 4u);
    keyed.latest.sample_tick_ms = 1000;
    keyed.scene_hold_until = {1500, 0};
    keyed.adopt(4u | 16u, 0u, 4u);
    require(keyed.latest.sample_tick_ms == 1000 && keyed.scene_hold_until[0] == 1500, "A HUD-less pairing dropped a sample or a hold");
    keyed.adopt(4u | 16u, 0u, 0u);
    require(!keyed.latest.sample_tick_ms && keyed.scene_hold_until[0] == 1500, "Losing acceptance kept the sample or cleared the route");
    keyed.adopt(0x40u, layer, 0u);
    require(!keyed.scene_hold_until[0], "A new route kept its hold");
    // A sample describes frames within 500 ms of its tick, and is discarded
    // under another scope or decision key.
    require(!temporal::sample_stale(1000, 1500) && temporal::sample_stale(1000, 1501) && temporal::sample_stale(0, 10),
      "The sample freshness bound moved");
    alpha_auto_source now, pending;
    now.now_ms = 1200; pending.now_ms = 1000;
    require(!temporal::sample_discarded(now, pending, 7, 7) && temporal::sample_discarded(now, pending, 7, 8),
      "A sample under the same inputs was discarded, or one under other inputs kept");
    ++now.revision;
    require(temporal::sample_discarded(now, pending, 7, 7), "A sample from another revision was kept");
    // One valid hidden sample holds the layer route for hold_ms from its tick;
    // a visible one releases it and refutes the input until it is an overlay.
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
    // An accepted layer opens no route.
    temporal::detection_state accepted_layer;
    auto decided = sample;
    decided.evidence.accepted = 0x40u; decided.evidence.scene.valid = true;
    accepted_layer.observe_scene(decided, true);
    require(!accepted_layer.scene_gate_open && !accepted_layer.scene_holds(2600), "An accepted layer opened the layer route");
    // A scope change and an inactive render clear the holds.
    scene.latest_source.epoch = 1;
    scene.enter_scope(now);
    require(!scene.scene_holds(2500), "A new scope kept a hold");
    scene.scene_hold_until = {3000, 3000}; scene.scene_refuted_slots = 2u; scene.mask_ready = scene.exact = true;
    scene.inactive();
    require(!scene.scene_holds(2500) && !scene.scene_refuted_slots && !scene.mask_ready && !scene.exact,
      "An inactive render kept a hold, refutation or holdable mask");
  }

  // Exact UI counters (game3d_ui_counters.h): the log text, the accounting
  // identity, wrap-safe GPU deltas, session totals and acceptance events.
  void ui_counters_are_formatted_and_trust_events_counted() {
    namespace n = ui_counter;
    ui_counters c;
    c[n::auto_frames] = 10; c[n::detection_frames] = 6;
    c[n::held_generated] = 2; c[n::held_inexact_after_exact] = 1; c[n::held_cap] = 1;
    c[n::inactive_no_candidates] = 1;
    c[n::decided + 0] = 2; c[n::decided + 5] = 3; c[n::decided + 6] = 1; c[n::decided + 10] = 4;
    c[n::none + ui_no_mask::difference_failed] = 1; c[n::none + ui_no_mask::gate_no_hold] = 1;
    c[n::none + ui_no_mask::unaccepted] = 2;
    c[n::depth_not_current] = 1; c[n::full_d_hidden] = 1; c[n::inexact_difference] = 3; c[n::trust_earned] = 1;
    c[n::trust_discarded] = 2;
    c[n::full_alpha] = 2; c[n::full_alpha_d_visible] = 1;
    c[n::samples] = 4; c.through_ms = 12345;
    require(c.reconciled() && c.held() == 3 && c.inactive() == 1, "The counters' accounting identity is wrong");
    require(format_ui_counters(c) ==
        "auto_frames=10 detection_frames=6 held={generated=2 inexact_after_exact=1 trusted_missing=0 cap=1} "
        "inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 9=0 10=4} "
        "none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 gate_no_hold=1 "
        "no_candidate=0 other=0 unaccepted=2} full={6=1 8=0 9=0 depth_not_current=1} full_d={hidden=1 ambiguous=0 visible=0 invalid=0} "
        "untrusted_inferred=0 inexact_difference=3 trusted_full=0 presented_over_dedicated=0 full_alpha=2 "
        "full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_full=0 "
        "revoked_presented=0 lapsed=0 restored=0 discarded=2} samples=4 through_ms=12345",
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
    ui_counters gpu;
    gpu.add_gpu_delta(now, before);
    require(gpu[n::detection_frames] == 3 && gpu.decided(4) == 2 && gpu.decided(10) == 6 &&
        gpu[n::none + ui_no_mask::ambiguous] == 5 && gpu[n::none + ui_no_mask::unaccepted] == 3 &&
        gpu[n::untrusted_inferred] == 2 && gpu[n::full_alpha] == 4 && !gpu.decided(0), "GPU counter deltas are wrong");
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

    // Earned once, however many samples confirm it; revoked by a full claim
    // over a visible scene.
    alpha_auto_policy earned;
    for (std::uint64_t tick = 10000; tick <= 13000; tick += 1000) feed(earned, sample().alpha(kind::backbuffer, 200), tick);
    require(accepts(earned, kind::backbuffer) && earned.counters()[n::trust_earned] == 1, "Earning was not counted once");
    for (std::uint64_t tick = 14000; tick <= 16000; tick += 1000)
      feed(earned, sample().alpha(kind::backbuffer, pixels).pair(100, 900), tick);
    require(!accepts(earned, kind::backbuffer) && earned.counters()[n::trust_revoked_full] == 1 &&
        !earned.counters()[n::trust_revoked_presented], "A full claim over a visible scene was not counted as its revocation");
    // Presented alpha disagreeing with an accepted UIAlpha.
    alpha_auto_policy presented;
    for (std::uint64_t tick = 10000; tick <= 12000; tick += 1000)
      feed(presented, sample().alpha(kind::ui_alpha, 100).alpha(kind::backbuffer, 100), tick);
    require(accepts(presented, kind::ui_alpha) && accepts(presented, kind::backbuffer) &&
        presented.counters()[n::trust_earned] == 2, "Two sources were not accepted");
    for (std::uint64_t tick = 13000; tick <= 15000; tick += 1000)
      feed(presented, sample().alpha(kind::ui_alpha, 100).alpha(kind::backbuffer, 300), tick);
    require(accepts(presented, kind::ui_alpha) && !accepts(presented, kind::backbuffer) &&
        presented.counters()[n::trust_revoked_presented] == 1, "A presented-alpha disagreement was not counted as its revocation");
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
    std::array<std::uint32_t, 4 * ui_detection::layer_decision_texels> t{};
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
    const auto sample = ui_temporal::decode_detection_sample(t.data(), t.size(), 1500, 9, true);
    const auto &e = sample.evidence;
    require(sample.source_kind == 10 && sample.enabled && sample.state == alpha_auto_state::automatic_on &&
        sample.covered == 995 && sample.pixels == 1000 && sample.sample_tick_ms == 1500 && sample.sample_sequence == 9 &&
        sample.accepted_samples == 9 && e.matching_tiles == 7 && e.candidates == 0x7a && e.hudless_changed == 9 &&
        e.hudless_unchanged == 900 && e.hudless_invalid == 1 && e.hudless_lit == 800 && e.accepted == 0x42 &&
        e.alpha_covered[3] == 13 && e.alpha_invalid[0] == 20 && e.alpha_opaque[1] == 31 && e.layer_covered == 995 &&
        e.layer_invalid == 2 && e.layer_opaque == 990 && e.valid_bits == 0x4a,
      "The decision texels did not decode");
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
    revocation_is_per_signature();
    exact_change_sets_earn();
    manual_on_is_a_session_override();
    only_the_deciding_inputs_key_a_status_sample();
    hidden_scene_gates_and_samples_that_flatten_learn_nothing();
    temporal_state_holds_and_scene_verdicts();
    ui_counters_are_formatted_and_trust_events_counted();
    sample_decode_and_commit_are_shared();
    std::puts("Source alpha session: 14 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Source alpha session failed: %s\n", error.what());
    return 1;
  }
}
