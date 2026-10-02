// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_alpha_auto.h"
#include "game3d_ui_detection_contract.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
  using namespace sunshine_game3d;

  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
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
  void alpha_trust_is_earned_by_selective_coverage_and_lost_on_contradiction() {
    const std::uint32_t pixels = 1000;
    const auto sample = [&](alpha_auto_policy &policy, std::uint32_t candidates, std::array<std::uint32_t, 4> covered,
        std::uint32_t unchanged, std::uint64_t tick, std::array<std::uint32_t, 4> invalid = {}) {
      alpha_auto_decision::detection_evidence evidence;
      evidence.candidates = candidates; evidence.alpha_covered = covered; evidence.alpha_invalid = invalid;
      evidence.hudless_unchanged = unchanged;
      policy.observe_alpha_channels(evidence, pixels, tick);
    };
    alpha_auto_policy policy;
    sample(policy, 4, {0, 0, 200, 0}, 0, 10000);
    sample(policy, 4, {0, 0, 200, 0}, 0, 11000);
    sample(policy, 4, {0, 0, 200, 0}, 0, 11999);
    require(!policy.trusted_alpha(), "Three samples within 2 s earned trust");
    sample(policy, 4, {0, 0, 200, 0}, 0, 12000);
    require(policy.trusted_alpha() == 4u, "Selective samples over 2 s did not earn exactly their channel");
    for (std::uint64_t tick = 13000; tick <= 20000; tick += 1000) {
      sample(policy, 2 | 8, {0, 1000, 0, 0}, 0, tick);        // Full-scene alpha.
      sample(policy, 8, {0, 0, 0, 0}, 0, tick);               // Empty alpha.
      sample(policy, 1, {0, 0, 0, 0}, 0, tick, {5, 0, 0, 0}); // Invalid alpha.
      sample(policy, 0, {300, 300, 300, 300}, 0, tick);       // Nothing offered.
    }
    require(policy.trusted_alpha() == 4u, "Empty, full-scene, invalid or unoffered alpha earned trust");
    // A full channel over a menu, or without an exact pair, is no contradiction.
    for (std::uint64_t tick = 21000; tick <= 25000; tick += 1000) {
      sample(policy, 4 | 16 | 32, {0, 0, 1000, 0}, 100, tick);
      sample(policy, 4 | 16, {0, 0, 1000, 0}, 900, tick);
      sample(policy, 4, {0, 0, 1000, 0}, 900, tick);
    }
    require(policy.trusted_alpha() == 4u, "A menu or an inexact pair revoked trust");
    // An exact pair showing 75% of the scene under a full channel revokes it,
    // after the same evidence interval; a selective sample restarts the doubt.
    sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, 30000);
    sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, 31000);
    sample(policy, 4, {0, 0, 200, 0}, 0, 31500);
    sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, 32000);
    sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, 33000);
    require(policy.trusted_alpha() == 4u, "Doubt survived a selective sample");
    sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, 34000);
    require(!policy.trusted_alpha(), "Contradiction over 2 s did not revoke trust");
    sample(policy, 4, {0, 0, 200, 0}, 0, 35000);
    sample(policy, 4, {0, 0, 200, 0}, 0, 36000);
    require(!policy.trusted_alpha(), "Revocation kept earlier selective evidence");
    sample(policy, 4, {0, 0, 200, 0}, 0, 37000);
    require(policy.trusted_alpha() == 4u, "A revoked channel could not earn trust again");
    policy.set_manual(false);
    policy.set_automatic();
    require(policy.trusted_alpha() == 4u, "Mode edits forgot what the game session trusted");
    alpha_auto_policy clock;
    sample(clock, 1, {100, 0, 0, 0}, 0, 5000);
    sample(clock, 1, {100, 0, 0, 0}, 0, 4000); // Earlier than the first: restart the interval.
    sample(clock, 1, {100, 0, 0, 0}, 0, 5500);
    sample(clock, 1, {100, 0, 0, 0}, 0, 5999);
    require(!clock.trusted_alpha(), "A backwards clock kept the earlier interval");
    sample(clock, 1, {100, 0, 0, 0}, 0, 6000);
    require(clock.trusted_alpha() == 1u, "The restarted interval did not earn trust");
    // Earning needs consecutive, steady samples: a full-frame sample restarts
    // the run, and coverage that swings beyond a factor of two (scene effects
    // such as particles in a transparent target) never earns trust.
    alpha_auto_policy run;
    sample(run, 1, {200, 0, 0, 0}, 0, 1000);
    sample(run, 1, {200, 0, 0, 0}, 0, 2000);
    sample(run, 1, {1000, 0, 0, 0}, 0, 2500);
    sample(run, 1, {200, 0, 0, 0}, 0, 3000);
    require(!run.trusted_alpha(), "A full-frame sample did not restart the earning run");
    sample(run, 1, {300, 0, 0, 0}, 0, 4000);
    sample(run, 1, {250, 0, 0, 0}, 0, 5000);
    require(run.trusted_alpha() == 1u, "Consecutive steady samples did not earn trust");
    alpha_auto_policy effects;
    for (std::uint64_t tick = 1000; tick <= 20000; tick += 500)
      sample(effects, 1, {tick % 1000 ? 50u : 400u, 0, 0, 0}, 0, tick);
    require(!effects.trusted_alpha(), "Fluctuating coverage earned trust");
    // A trusted dedicated UI channel contradicts presented alpha that differs
    // from it by 10% of the frame or more (Resident Evil Requiem: UI color 0.2%,
    // presented alpha 35-100% during play; in menus UI color 100%, presented
    // alpha 0.05%). Agreement is no contradiction, and without a trusted
    // dedicated channel the presented alpha keeps its trust.
    alpha_auto_policy presented;
    presented.restore_trusted_alpha(2 | 8);
    for (std::uint64_t tick = 1000; tick <= 4000; tick += 1000)
      sample(presented, 2 | 8, {0, 1000, 0, 950}, 0, tick);    // Menu: both cover the screen.
    require(presented.trusted_alpha() == 10u, "Presented alpha agreeing with a full-screen menu was contradicted");
    sample(presented, 2 | 8, {0, 2, 0, 999}, 0, 5000);
    sample(presented, 2 | 8, {0, 0, 0, 400}, 0, 6000);
    require(presented.trusted_alpha() == 10u, "Two samples revoked trust");
    sample(presented, 2 | 8, {0, 2, 0, 350}, 0, 7000);
    require(presented.trusted_alpha() == 2u, "A dedicated UI mask did not revoke presented alpha that covers the scene");
    sample(presented, 2 | 8, {0, 2, 0, 300}, 0, 8000);
    sample(presented, 2 | 8, {0, 2, 0, 300}, 0, 9000);
    sample(presented, 2 | 8, {0, 2, 0, 300}, 0, 10000);
    require(presented.trusted_alpha() == 2u, "Contradicted presented alpha earned trust again");
    // Menu: the UI channel covers the screen, presented alpha nearly nothing.
    for (std::uint64_t tick = 10500; tick <= 13500; tick += 500) sample(presented, 2 | 8, {0, 1000, 0, 4}, 0, tick);
    require(presented.trusted_alpha() == 2u, "Presented alpha re-earned trust in a menu its UI channel covers");
    sample(presented, 2 | 8, {0, 20, 0, 30}, 0, 14000);
    sample(presented, 2 | 8, {0, 20, 0, 30}, 0, 15000);
    sample(presented, 2 | 8, {0, 20, 0, 30}, 0, 16000);
    require(presented.trusted_alpha() == 10u, "Presented alpha agreeing with the UI mask could not earn trust");
    // Remembered trust lapses unless the session earns it again: a channel that
    // only ever covers the scene unsteadily loses it after a minute of samples,
    // while one that earns it again keeps it for good.
    alpha_auto_policy remembered;
    remembered.restore_trusted_alpha(8);
    for (std::uint64_t tick = 1000; tick <= 60000; tick += 1000)
      sample(remembered, 8, {0, 0, 0, tick % 2000 ? 500u : 50u}, 0, tick);
    require(remembered.trusted_alpha() == 8u, "Remembered trust lapsed before a minute");
    sample(remembered, 8, {0, 0, 0, 50}, 0, 61000);
    require(!remembered.trusted_alpha(), "Unconfirmed remembered trust did not lapse");
    alpha_auto_policy confirmed;
    confirmed.restore_trusted_alpha(1);
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) sample(confirmed, 1, {100, 0, 0, 0}, 0, tick);
    for (std::uint64_t tick = 4000; tick <= 200000; tick += 1000) sample(confirmed, 1, {0, 0, 0, 0}, 0, tick);
    require(confirmed.trusted_alpha() == 1u, "Trust earned again in this session lapsed later");
    // An untrusted UI channel (here unsteady, so it never earns trust) is no authority.
    alpha_auto_policy alone;
    alone.restore_trusted_alpha(8);
    for (std::uint64_t tick = 1000; tick <= 8000; tick += 1000)
      sample(alone, 2 | 8, {0, tick % 2000 ? 2u : 40u, 0, 400}, 0, tick);
    require(alone.trusted_alpha() == 8u, "An untrusted UI channel revoked presented alpha");
  }
  void remembered_alpha_trust_is_restored_and_every_change_is_reported() {
    const std::uint32_t pixels = 1000;
    const auto sample = [&](alpha_auto_policy &policy, std::uint32_t candidates, std::array<std::uint32_t, 4> covered,
        std::uint32_t unchanged, std::uint64_t tick) {
      alpha_auto_decision::detection_evidence evidence;
      evidence.candidates = candidates; evidence.alpha_covered = covered; evidence.hudless_unchanged = unchanged;
      policy.observe_alpha_channels(evidence, pixels, tick);
    };
    alpha_auto_policy policy;
    std::vector<std::uint32_t> reported;
    policy.on_trust_change([&](std::uint32_t trusted) { reported.push_back(trusted); });
    policy.restore_trusted_alpha(4u | 64u);
    require(policy.trusted_alpha() == 4u, "A remembered channel was not trusted at once, or unknown bits were kept");
    require(reported.empty(), "Restoring remembered trust was reported as a change");
    // A title screen or menu before any selective frame is already covered.
    sample(policy, 4, {0, 0, 1000, 0}, 0, 1000);
    sample(policy, 4, {0, 0, 200, 0}, 0, 1500);
    require(policy.trusted_alpha() == 4u && reported.empty(), "Samples of a remembered channel changed or reported trust");
    // Earning another channel reports the combined trust once.
    for (const std::uint64_t tick : {2000u, 3000u, 4000u, 5000u})
      sample(policy, 1 | 4, {200, 0, 200, 0}, 0, tick);
    require(policy.trusted_alpha() == 5u, "A second channel did not earn trust beside the remembered one");
    require(reported == std::vector<std::uint32_t> {5u}, "Earning was not reported exactly once");
    // The same contradiction that revokes earned trust revokes remembered trust.
    for (const std::uint64_t tick : {10000u, 11000u, 12000u})
      sample(policy, 4 | 48, {0, 0, 1000, 0}, 750, tick);
    require(policy.trusted_alpha() == 1u, "Contradiction did not revoke a remembered channel");
    require(reported == std::vector<std::uint32_t> {5u, 1u}, "Revocation was not reported for persistence");
  }

  void trust_belongs_to_the_source_that_filled_the_slot() {
    // Slot 1 holds either the tagged UI color+alpha (source 1) or the offscreen
    // UI layer (source 4). Stellar Blade offers the layer with frame generation
    // off and an opaque tagged UI color with it on; the tag must not inherit the
    // layer's trust, and its contradiction must not revoke the layer.
    const std::uint32_t pixels = 1000;
    const auto sample = [&](alpha_auto_policy &policy, bool layer, std::uint32_t candidates,
        std::array<std::uint32_t, 4> covered, std::uint32_t unchanged, std::uint64_t tick) {
      alpha_auto_decision::detection_evidence evidence;
      evidence.candidates = candidates; evidence.alpha_covered = covered; evidence.hudless_unchanged = unchanged;
      evidence.ui_layer = layer;
      policy.observe_alpha_channels(evidence, pixels, tick);
    };
    alpha_auto_policy policy;
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) sample(policy, true, 2 | 8, {0, 20, 0, 1000}, 0, tick);
    require(policy.trusted_alpha() == 16u, "The UI layer did not earn trust as its own source");
    require(policy.trusted_slots(true) == 2u && !policy.trusted_slots(false), "Slot trust did not follow the slot's source");
    require(policy.prefer_ui_layer(), "A trusted UI layer was not preferred over an untrusted tag");
    // The opaque tag over a visible scene (exact HUD-less pair, 80% unchanged)
    // earns nothing and revokes nothing.
    for (std::uint64_t tick = 4000; tick <= 10000; tick += 1000)
      sample(policy, false, 2 | 4 | 48, {0, 1000, 1000, 0}, 800, tick);
    require(policy.trusted_alpha() == 16u, "The opaque tag revoked the UI layer or earned trust");
    // A trusted UI layer is the dedicated mask that contradicts presented alpha.
    policy.restore_trusted_alpha(8);
    for (std::uint64_t tick = 11000; tick <= 13000; tick += 1000)
      sample(policy, true, 2 | 8, {0, 20, 0, 400}, 0, tick);
    require(policy.trusted_alpha() == 16u, "A trusted UI layer did not revoke disagreeing presented alpha");

    // Without layer trust, a tag that proves opaque over a visible scene hands
    // its slot to the layer; a selective tag sample takes the slot back.
    alpha_auto_policy opaque;
    sample(opaque, false, 2 | 48, {0, 1000, 0, 0}, 600, 1000);
    sample(opaque, false, 2 | 48, {0, 1000, 0, 0}, 600, 2000);
    require(!opaque.prefer_ui_layer(), "Two samples proved the tag opaque");
    sample(opaque, false, 2 | 16, {0, 1000, 0, 0}, 600, 2500); // Inexact pair: no evidence either way.
    sample(opaque, false, 2 | 48, {0, 1000, 0, 0}, 600, 3000);
    require(opaque.prefer_ui_layer(), "An opaque tag over a visible scene was still preferred");
    sample(opaque, false, 2 | 48, {0, 30, 0, 0}, 600, 4000);
    require(!opaque.prefer_ui_layer(), "A selective tag sample did not restore the tag");
    // A full-screen menu (HUD-less differs almost everywhere) proves nothing.
    alpha_auto_policy menu;
    for (std::uint64_t tick = 1000; tick <= 5000; tick += 1000) sample(menu, false, 2 | 48, {0, 1000, 0, 0}, 50, tick);
    require(!menu.prefer_ui_layer(), "A full-screen menu proved the tag opaque");
    // A trusted tag keeps its slot even beside a trusted layer.
    alpha_auto_policy both;
    both.restore_trusted_alpha(2 | 16);
    require(!both.prefer_ui_layer() && both.trusted_slots(false) == 2u && both.trusted_slots(true) == 2u,
      "A trusted tag lost its slot");
    static_assert(alpha_auto_policy::sources_from_slots(1 | 2 | 4 | 8) == (1 | 4 | 8),
      "Remembered slot trust kept the ambiguous UI color slot");
  }
  void only_the_deciding_inputs_key_a_status_sample() {
    // Resident Evil Requiem with FG: trusted UI color alpha (2) and present
    // alpha (8); its HUD-less pair (16) joins on some Presents only.
    const std::uint32_t trusted = 2u | 8u;
    require(detection_decision_key(2u, 0u, trusted) == detection_decision_key(2u | 16u, 0u, trusted),
      "A HUD-less pair beside a trusted channel changed the decision");
    require(detection_decision_key(2u | 8u, 0u, trusted) == detection_decision_key(2u, 0u, trusted),
      "A later trusted channel beside the first one changed the decision");
    require(detection_decision_key(8u, 0u, trusted) != detection_decision_key(2u, 0u, trusted),
      "A different deciding channel kept the decision");
    require(detection_decision_key(2u, 1u, trusted) != detection_decision_key(2u, 0u, trusted),
      "Flags of the deciding UI color channel were ignored");
    require(detection_decision_key(1u | 2u, 1u, 1u) == detection_decision_key(1u | 2u, 0u, 1u),
      "UI color flags changed a decision another channel makes");
    // The offscreen UI layer's stored flags (5 for 8 bits) key its decision
    // apart from a tagged UIColorAndAlpha (0), and the same layer keeps its
    // key frame after frame.
    const auto layer = ui_detection::layer_detection_flags(false);
    require(detection_decision_key(2u, layer, 0u) != detection_decision_key(2u, 0u, 0u),
      "The UI layer and a tagged UI color shared a decision key");
    require(detection_decision_key(2u, layer, 0u) == detection_decision_key(2u, 5u, 0u) &&
        detection_decision_key(2u | 8u, layer, 2u) == detection_decision_key(2u, 5u, 2u),
      "A repeated layer frame changed its decision key");
    require(detection_decision_key(1u | 2u, layer, 1u) == detection_decision_key(1u | 2u, 0u, 1u),
      "Layer flags changed a decision the trusted UI alpha channel makes");
    require(detection_decision_key(2u, layer | ui_detection::stored_hdr_headroom, 0u) != detection_decision_key(2u, layer, 0u),
      "A float layer kept an 8-bit layer's decision");
    // Without trust every candidate, exactness and trust itself can matter.
    require(detection_decision_key(2u, 0u, 0u) != detection_decision_key(2u | 16u, 0u, 0u),
      "A HUD-less candidate without a trusted channel kept the decision");
    require(detection_decision_key(16u, 0u, 0u) != detection_decision_key(48u, 0u, 0u),
      "An exact HUD-less pair kept the inexact decision");
    require(detection_decision_key(2u | 16u, 0u, trusted) != detection_decision_key(2u | 16u, 0u, 0u),
      "Losing trust kept the decision");
  }
} // namespace

int main() {
  try {
    mode_is_auto_until_a_manual_edit_and_auto_again_after_it();
    alpha_trust_is_earned_by_selective_coverage_and_lost_on_contradiction();
    remembered_alpha_trust_is_restored_and_every_change_is_reported();
    trust_belongs_to_the_source_that_filled_the_slot();
    only_the_deciding_inputs_key_a_status_sample();
    std::puts("Source alpha session: 5 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Source alpha session failed: %s\n", error.what());
    return 1;
  }
}
