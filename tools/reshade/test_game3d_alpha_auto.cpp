// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_alpha_auto.h"
#include "game3d_ui_detection_contract.h"
#include "game3d_ui_temporal.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
    // A trusted layer without alpha is no dedicated mask: it neither revokes
    // nor shields selective presented alpha, which earns trust over 2 s, and
    // its own trust is untouched.
    alpha_auto_policy aside;
    aside.restore_trusted_alpha(16);
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) {
      alpha_auto_decision::detection_evidence evidence;
      evidence.candidates = 2 | 4; evidence.ui_layer = true;
      evidence.alpha_covered = {0, 0, 3, 0}; evidence.alpha_invalid = {0, 600, 0, 0};
      aside.observe_alpha_channels(evidence, pixels, tick);
    }
    require(aside.trusted_alpha() == (16u | 4u), "A layer without alpha shielded or revoked selective presented alpha");
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
        detection_decision_key(2u | 8u, layer, 2u) == detection_decision_key(2u | 8u | 16u, layer, 2u) &&
        detection_decision_key(2u | 8u | 16u, layer, 2u) == detection_decision_key(2u | 8u | 48u, layer, 2u),
      "A repeated layer frame changed its decision key");
    // A trusted layer does not decide alone: a layer without alpha is set aside
    // by its own pixels, and presented alpha then decides by its trust. Its key
    // holds the alpha channels' trust, never which presented channels a Present
    // offers (Stellar Blade with FG tags the Backbuffer on real Presents only;
    // a key that changed with it starved the generated-Present hold of the
    // sample it reads) nor a HUD-less pairing.
    require(detection_decision_key(2u | 4u, layer, 2u | 4u) != detection_decision_key(4u, 0u, 2u | 4u) &&
        detection_decision_key(2u | 4u, layer, 2u | 4u) != detection_decision_key(4u, 0u, 4u),
      "A trusted layer beside the Backbuffer shared the Backbuffer's decision key");
    require(detection_decision_key(2u | 4u, layer, 2u | 4u) != detection_decision_key(2u, layer, 2u) &&
        detection_decision_key(2u | 8u, layer, 2u) != detection_decision_key(2u | 8u, layer, 2u | 8u),
      "Presented alpha's trust beside a trusted layer kept the layer-alone decision key");
    require(detection_decision_key(2u | 4u, layer, 2u | 4u) == detection_decision_key(2u, layer, 2u | 4u) &&
        detection_decision_key(2u | 8u, layer, 2u) == detection_decision_key(2u, layer, 2u),
      "A presented channel offered on some Presents only changed a trusted layer's decision key");
    require(detection_decision_key(2u | 4u | 16u, layer, 2u | 4u) == detection_decision_key(2u | 4u | 48u, layer, 2u | 4u) &&
        detection_decision_key(2u | 4u, layer, 2u | 4u) == detection_decision_key(2u | 4u | 48u, layer, 2u | 4u),
      "A HUD-less pairing beside a trusted layer changed the decision key");
    require(detection_decision_key(2u | 4u, 0u, 2u | 4u) == detection_decision_key(2u, 0u, 2u) &&
        detection_decision_key(2u | 4u | 16u, 0u, 2u | 4u) == detection_decision_key(2u, 0u, 2u | 4u),
      "A trusted tagged UI color stopped deciding alone");
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
  // Hidden-scene gates (docs/reshade-sbs.md, hidden-scene evidence) open only
  // when no source 1-6 decided, for an untrusted UIAlpha or offscreen UI layer
  // nearly opaque without an invalid pixel, or a HUD-less image differing on
  // at least 90% of the frame. A sample that decided 8 or 9 still shows them.
  void hidden_scene_gates_and_samples_that_flatten_learn_nothing() {
    namespace detection = ui_detection;
    const std::uint32_t pixels = 1000;
    const std::array<std::uint32_t, 4> clean{};
    const auto gates = [&](std::uint32_t source, std::uint32_t candidates, std::uint32_t trusted, std::uint32_t flags,
        std::array<std::uint32_t, 2> opaque, std::uint32_t changed, std::array<std::uint32_t, 4> invalid = {}) {
      return detection::scene_gates_of(source, candidates, trusted, flags, pixels, invalid, opaque, changed);
    };
    const auto layer = detection::layer_detection_flags(false);
    require(gates(0, 2, 0, layer, {0, 990}, 0).layer && gates(8, 2, 0, layer, {0, 1000}, 0).layer,
      "An untrusted opaque layer did not open the layer route");
    require(!gates(0, 2, 0, layer, {0, 989}, 0).layer, "A layer under 99% opaque opened the layer route");
    require(!gates(0, 2, 0, 0u, {0, 1000}, 0).layer, "A tagged UI color opened the layer route");
    require(!gates(0, 2, 2, layer, {0, 1000}, 0).layer, "A trusted layer opened the layer route");
    require(!gates(0, 2, 0, layer, {0, 1000}, 0, {0, 1, 0, 0}).layer, "A layer with an invalid pixel opened the layer route");
    require(gates(0, 1, 0, 0u, {1000, 0}, 0).layer && !gates(0, 1, 1, 0u, {1000, 0}, 0).layer,
      "UIAlpha did not open the layer route only while untrusted");
    require(!gates(0, 4 | 8, 0, 0u, {1000, 1000}, 0).layer, "Presented alpha opened the layer route");
    for (const std::uint32_t decided : {1u, 2u, 3u, 4u, 5u, 6u})
      require(!gates(decided, 2 | 16, 0, layer, {0, 1000}, 1000).layer && !gates(decided, 16, 0, 0u, {}, 1000).hudless,
        "A decided frame opened a hidden-scene route");
    require(gates(0, 16, 0, 0u, {}, 900).hudless && gates(9, 48, 0, 0u, {}, 1000).hudless && !gates(0, 16, 0, 0u, {}, 899).hudless &&
        !gates(0, 0, 0, 0u, {}, 1000).hudless, "The HUD-less route did not open at 90% changed pixels exactly");
    require(detection::scene_visible(.25f) && !detection::scene_visible(.2499f), "The visible bound moved");
    // Per slot: the slots that opened the layer route, and the route's inputs
    // offered below that opacity, which shows them to be overlays (a slot a
    // visible verdict refuted needs one), whatever source decided.
    require(gates(0, 3, 0, layer, {1000, 989}, 0).layer_slots == 1u && gates(0, 3, 0, layer, {1000, 989}, 0).overlay_slots == 2u,
      "The layer route's slots were not told apart");
    require(gates(2, 2, 0, layer, {0, 100}, 0).overlay_slots == 2u && !gates(2, 2, 0, layer, {0, 100}, 0).layer_slots,
      "A layer that decided its own selective mask was not shown an overlay");
    require(!gates(0, 2, 0, 0u, {0, 100}, 0).overlay_slots && !gates(0, 4 | 8, 0, 0u, {0, 0}, 0).overlay_slots,
      "A tagged UI color or presented alpha counted as a layer-route input");
    // A layer without alpha (Stellar Blade's SDR scene image: no alpha, color
    // on 60% of pixels) is no layer: fed the admitted candidates, it is no
    // overlay, so it cannot clear a refuted slot.
    const std::array<std::uint32_t, 4> color_only{0, 600, 0, 0};
    const auto admitted = detection::admitted_candidates(2 | 4, layer, 0, color_only[1], pixels);
    require(admitted == 4u && gates(0, 2 | 4, 0, layer, {0, 0}, 0, color_only).overlay_slots == 2u &&
        !gates(0, admitted, 0, layer, {0, 0}, 0, color_only).overlay_slots,
      "A layer without alpha counted as an overlay");
    require(detection::admitted_candidates(2 | 4, layer, 0, 10, pixels) == (2u | 4u) &&
        detection::admitted_candidates(2 | 4, layer, 1, 600, pixels) == (2u | 4u) &&
        detection::admitted_candidates(2 | 4, 0u, 0, 600, pixels) == (2u | 4u),
      "A layer with alpha, at most 1% invalid, or a tagged UI color was set aside");
    // Holds and refutations follow the routes' inputs, not a HUD-less pairing.
    require(detection::scene_route_key(2, layer, 0) == detection::scene_route_key(2 | 16, layer, 0) &&
        detection::scene_route_key(2 | 16, layer, 0) == detection::scene_route_key(2 | 48, layer, 0) &&
        detection::scene_route_key(2 | 4 | 8, layer, 0) == detection::scene_route_key(2, layer, 0),
      "A HUD-less pairing or presented alpha changed the routes' inputs");
    require(detection::scene_route_key(2, layer, 0) != detection::scene_route_key(2, layer, 2) &&
        detection::scene_route_key(2, layer, 0) != detection::scene_route_key(2, 0u, 0) &&
        detection::scene_route_key(2, layer, 0) != detection::scene_route_key(0, 0u, 0) &&
        detection::scene_route_key(1, 0u, 0) != detection::scene_route_key(1, 0u, 1),
      "Trust, a slot's source or its stored flags did not change the routes' inputs");

    // The counts of a sample that decided 8 or 9 cover the whole frame with an
    // untrusted channel: they never earn or revoke trust and never prove a
    // tagged UI color opaque, so the layer is still preferred only by trust.
    alpha_auto_policy policy;
    alpha_auto_decision::detection_evidence evidence;
    evidence.candidates = 2 | 16; evidence.ui_layer = true;
    evidence.alpha_covered = {0, pixels, 0, 0}; evidence.alpha_opaque = {0, pixels};
    evidence.hudless_changed = pixels;
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 100) policy.observe_alpha_channels(evidence, pixels, tick);
    evidence.candidates = 1 | 8 | 48; evidence.ui_layer = false; evidence.alpha_covered = {pixels, 0, 0, pixels};
    for (std::uint64_t tick = 7000; tick <= 12000; tick += 100) policy.observe_alpha_channels(evidence, pixels, tick);
    require(!policy.trusted_alpha() && !policy.prefer_ui_layer(), "A full-frame hidden-scene sample earned trust or proved a UI color opaque");
    alpha_auto_policy trusting;
    trusting.restore_trusted_alpha(1u << alpha_auto_policy::ui_layer_source);
    evidence.candidates = 2 | 16; evidence.ui_layer = true; evidence.alpha_covered = {0, pixels, 0, 0};
    for (std::uint64_t tick = 1000; tick <= 6000; tick += 100) trusting.observe_alpha_channels(evidence, pixels, tick);
    require(trusting.trusted_alpha() == 1u << alpha_auto_policy::ui_layer_source,
      "A full-frame sample without a visible scene revoked trust");
    // Only a session without remembered trust runs the first-run shadow; the
    // caller says which, and nothing learned later changes it.
    alpha_auto_policy first;
    require(!first.first_run(), "A policy started as a first run without its owner saying so");
    first.set_first_run(true);
    evidence.candidates = 2; evidence.alpha_covered = {0, 100, 0, 0};
    for (std::uint64_t tick = 1000; tick <= 4000; tick += 500) first.observe_alpha_channels(evidence, pixels, tick);
    require(first.first_run() && first.trusted_alpha(), "Earning trust ended the first-run shadow");
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
    require(!state.arbitrate(0u, 0u, false, true, max_held).hold, "A generated Present held a mask no decision made");
    // A HUD-less decision makes the mask holdable; a generated Present then
    // holds it for at most max_held Presents.
    state.adopt(16u | 32u, 0u, 0u);
    state.detected();
    require(state.mask_ready && state.exact, "An exact HUD-less decision was not holdable");
    for (std::uint32_t i = 0; i != max_held; ++i) {
      const auto hold = state.arbitrate(0u, 0u, false, true, max_held);
      require(hold.hold && hold.kind == hold_kind::generated && !hold.cap_reached, "A generated Present did not hold");
      state.held();
    }
    const auto capped = state.arbitrate(0u, 0u, false, true, max_held);
    require(!capped.hold && capped.cap_reached && capped.kind == hold_kind::generated, "A hold past the cap was not reported");
    // A fresh decision restarts the count; an inexact pair right after an
    // exact decision holds, and a generated Present names the hold first.
    state.detected();
    require(!state.holds, "A fresh decision kept the hold count");
    const auto inexact = state.arbitrate(16u, 0u, false, false, max_held);
    require(inexact.hold && inexact.kind == hold_kind::inexact_after_exact, "An inexact pair after an exact one did not hold");
    require(state.arbitrate(16u, 0u, false, true, max_held).kind == hold_kind::generated, "Hold kinds lost their priority");
    require(!state.arbitrate(48u, 0u, false, false, max_held).hold, "An exact pair was held");
    // A trusted channel that decided and is missing holds; offered, it decides.
    temporal::detection_state trusted;
    trusted.adopt(4u, 0u, 4u);
    trusted.detected();
    require(trusted.mask_ready && !trusted.exact, "A trusted Backbuffer decision was not holdable");
    const auto missing = trusted.arbitrate(0u, 4u, false, false, max_held);
    require(missing.hold && missing.kind == hold_kind::trusted_missing, "A missing trusted channel was not held");
    require(!trusted.arbitrate(4u, 4u, false, true, max_held).hold, "A trusted channel in this frame did not decide by itself");
    // Untrusted alpha never makes a holdable mask.
    temporal::detection_state untrusted;
    untrusted.adopt(4u, 0u, 0u);
    untrusted.detected();
    require(!untrusted.mask_ready && !untrusted.arbitrate(0u, 0u, false, true, max_held).hold, "An untrusted mask was held");
    // A trusted layer the latest sample set aside does not stop the hold of
    // the Backbuffer's mask.
    const auto layer = ui_detection::layer_detection_flags(false);
    temporal::detection_state aside;
    aside.adopt(2u | 4u, layer, 2u | 4u);
    aside.detected();
    aside.latest.pixels = 1000; aside.latest.evidence.ui_layer = true; aside.latest.evidence.alpha_invalid = {0, 600, 0, 0};
    require(aside.layer_slot(2u, layer) && aside.layer_slot(0u, 0u) && !aside.layer_slot(2u, 0u), "The UI color slot's source was lost");
    require(aside.arbitrate(2u, 2u | 4u, true, true, max_held).hold, "A set-aside trusted layer stopped the hold");
    aside.latest.evidence.alpha_invalid = {};
    require(!aside.arbitrate(2u, 2u | 4u, true, true, max_held).hold, "A trusted layer with alpha did not decide by itself");
    // A change of the deciding inputs drops the sample; a HUD-less pairing
    // beside a trusted channel does not, and neither changes the route.
    temporal::detection_state keyed;
    keyed.adopt(4u, 0u, 4u);
    keyed.latest.sample_tick_ms = 1000;
    keyed.scene_hold_until = {1500, 0};
    keyed.adopt(4u | 16u, 0u, 4u);
    require(keyed.latest.sample_tick_ms == 1000 && keyed.scene_hold_until[0] == 1500, "A HUD-less pairing dropped a sample or a hold");
    keyed.adopt(4u | 16u, 0u, 0u);
    require(!keyed.latest.sample_tick_ms && keyed.scene_hold_until[0] == 1500, "Losing trust kept the sample or cleared the route");
    keyed.adopt(2u, layer, 0u);
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
    // a visible one releases it and refutes the slot until it is an overlay.
    temporal::detection_state scene;
    scene.adopt(2u, layer, 0u);
    alpha_auto_decision sample;
    sample.pixels = 1000; sample.sample_tick_ms = 2000;
    sample.evidence.candidates = 2u; sample.evidence.ui_layer = true; sample.evidence.alpha_opaque = {0, 1000};
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
    require(!scene.scene_holds(2400), "A refuted slot held the layer route");
    sample.evidence.alpha_opaque = {0, 500};
    scene.observe_scene(sample, true);
    require(!scene.scene_refuted_slots, "A slot offered as an overlay stayed refuted");
    sample.evidence.alpha_opaque = {0, 1000}; sample.sample_tick_ms = 2400;
    scene.observe_scene(sample, true);
    require(scene.scene_holds(2900), "A slot shown an overlay could not hold the route again");
    sample.sample_tick_ms = 2600; sample.evidence.scene.valid = false;
    scene.observe_scene(sample, true);
    require(scene.scene_holds(2900) && !scene.scene_holds(2901), "Invalid evidence renewed or released a hold");
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
  // identity, wrap-safe GPU deltas, session totals and trust events.
  void ui_counters_are_formatted_and_trust_events_counted() {
    namespace n = ui_counter;
    ui_counters c;
    c[n::auto_frames] = 10; c[n::detection_frames] = 6;
    c[n::held_generated] = 2; c[n::held_inexact_after_exact] = 1; c[n::held_cap] = 1;
    c[n::inactive_no_candidates] = 1;
    c[n::decided + 0] = 2; c[n::decided + 5] = 3; c[n::decided + 6] = 1;
    c[n::none + ui_no_mask::difference_failed] = 1; c[n::none + ui_no_mask::gate_no_hold] = 1;
    c[n::depth_not_current] = 1; c[n::full_d_hidden] = 1; c[n::inexact_difference] = 3; c[n::trust_earned] = 1;
    c[n::full_alpha] = 2; c[n::full_alpha_d_visible] = 1;
    c[n::samples] = 4; c.through_ms = 12345;
    require(c.reconciled() && c.held() == 3 && c.inactive() == 1, "The counters' accounting identity is wrong");
    require(format_ui_counters(c) ==
        "auto_frames=10 detection_frames=6 held={generated=2 inexact_after_exact=1 trusted_missing=0 cap=1} "
        "inactive={no_candidates=1 size=0 unprepared=0} decided={0=2 1=0 2=0 3=0 4=0 5=3 6=1 8=0 9=0} "
        "none={layer_aside=0 trusted_invalid=0 presented_blocked=0 ambiguous=0 difference_failed=1 gate_no_hold=1 "
        "no_candidate=0 other=0} full={6=1 8=0 9=0 depth_not_current=1} full_d={hidden=1 ambiguous=0 visible=0 invalid=0} "
        "untrusted_inferred=0 inexact_difference=3 trusted_full=0 presented_over_dedicated=0 full_alpha=2 "
        "full_alpha_d={hidden=0 ambiguous=0 visible=1 invalid=0} trust={earned=1 revoked_full=0 "
        "revoked_presented=0 lapsed=0 restored=0 opaque_set=0 opaque_cleared=0} samples=4 through_ms=12345",
      "The UI counters log text changed");
    ++c[n::auto_frames];
    require(!c.reconciled(), "An unaccounted frame reconciled");
    // GPU words are uint32 totals: their change survives a wrap.
    std::array<std::uint32_t, ui_counter_word::count> before{}, now{};
    before[ui_counter_word::detection_frames] = 0xfffffffeu; now[ui_counter_word::detection_frames] = 1u;
    before[ui_counter_word::decided + 4] = 7u; now[ui_counter_word::decided + 4] = 9u;
    now[ui_counter_word::none + ui_no_mask::ambiguous] = 5u; now[ui_counter_word::untrusted_inferred] = 2u;
    now[ui_counter_word::full_alpha] = 4u;
    ui_counters gpu;
    gpu.add_gpu_delta(now, before);
    require(gpu[n::detection_frames] == 3 && gpu.decided(4) == 2 && gpu[n::none + ui_no_mask::ambiguous] == 5 &&
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

    const std::uint32_t pixels = 1000;
    const auto sample = [&](alpha_auto_policy &policy, std::uint32_t candidates, std::array<std::uint32_t, 4> covered,
        std::uint32_t unchanged, std::uint64_t tick) {
      alpha_auto_decision::detection_evidence evidence;
      evidence.candidates = candidates; evidence.alpha_covered = covered; evidence.hudless_unchanged = unchanged;
      policy.observe_alpha_channels(evidence, pixels, tick);
    };
    // Earned once, however many samples confirm it; revoked by a full claim
    // over a visible scene.
    alpha_auto_policy earned;
    for (std::uint64_t tick = 10000; tick <= 13000; tick += 1000) sample(earned, 4, {0, 0, 200, 0}, 0, tick);
    require(earned.trusted_alpha() == 4u && earned.counters()[n::trust_earned] == 1, "Earning trust was not counted once");
    for (std::uint64_t tick = 14000; tick <= 16000; tick += 1000) sample(earned, 4 | 16 | 32, {0, 0, pixels, 0}, 900, tick);
    require(!earned.trusted_alpha() && earned.counters()[n::trust_revoked_full] == 1 &&
        !earned.counters()[n::trust_revoked_presented], "A full claim over a visible scene was not counted as its revocation");
    // Presented alpha disagreeing with a trusted UI alpha channel.
    alpha_auto_policy presented;
    for (std::uint64_t tick = 10000; tick <= 12000; tick += 1000) sample(presented, 1 | 4, {100, 0, 100, 0}, 0, tick);
    require(presented.trusted_alpha() == 5u && presented.counters()[n::trust_earned] == 2, "Two sources did not earn trust");
    for (std::uint64_t tick = 13000; tick <= 15000; tick += 1000) sample(presented, 1 | 4, {100, 0, 300, 0}, 0, tick);
    require(presented.trusted_alpha() == 1u && presented.counters()[n::trust_revoked_presented] == 1,
      "A presented-alpha disagreement was not counted as its revocation");
    // Restored trust lapses unless this session earns it again, and an earn
    // that confirms it counts.
    alpha_auto_policy lapsing;
    lapsing.restore_trusted_alpha(4u);
    sample(lapsing, 4, {0, 0, pixels, 0}, 0, 1000);
    sample(lapsing, 4, {0, 0, pixels, 0}, 0, 61000);
    require(!lapsing.trusted_alpha() && lapsing.counters()[n::trust_restored] == 1 && lapsing.counters()[n::trust_lapsed] == 1,
      "A restore or a provisional lapse was not counted");
    alpha_auto_policy confirmed;
    confirmed.restore_trusted_alpha(4u | 8u);
    for (std::uint64_t tick = 1000; tick <= 3000; tick += 1000) sample(confirmed, 4, {0, 0, 200, 0}, 0, tick);
    require(confirmed.counters()[n::trust_restored] == 2 && confirmed.counters()[n::trust_earned] == 1,
      "Confirming a restored source was not counted as earned");
    // The tagged UI color's opaque proof, set once and cleared by a selective sample.
    alpha_auto_policy opaque;
    for (std::uint64_t tick = 1000; tick <= 4000; tick += 1000) sample(opaque, 2 | 16 | 32, {0, pixels, 0, 0}, 600, tick);
    require(opaque.prefer_ui_layer() && opaque.counters()[n::trust_opaque_set] == 1, "The opaque proof was not counted once");
    sample(opaque, 2, {0, 100, 0, 0}, 0, 5000);
    require(!opaque.prefer_ui_layer() && opaque.counters()[n::trust_opaque_cleared] == 1, "Clearing the opaque proof was not counted");
  }
  // The pure pieces the renderer and the sequence replay share
  // (game3d_ui_temporal.h): the decode of a sample's decision texels, the
  // counts it commits and when a sample frame measures the scene.
  void sample_decode_and_commit_are_shared() {
    namespace word = ui_detection::decision_word;
    std::array<std::uint32_t, 4 * ui_detection::scene_decision_texels> t{};
    t[word::source] = 2; t[word::covered] = 995; t[word::pixels] = 1000; t[word::matching_tiles] = 7;
    t[word::candidates] = 0x3a; t[word::hudless_changed] = 9; t[word::hudless_unchanged] = 900;
    t[word::hudless_invalid] = 1; t[word::hudless_lit] = 800; t[word::trusted] = 2;
    for (std::uint32_t i = 0; i != 4; ++i) { t[word::alpha_covered + i] = 10 + i; t[word::alpha_invalid + i] = 20 + i; }
    t[word::alpha_opaque] = 30; t[word::alpha_opaque + 1] = 31;
    const float d = .5f, hudless_d = .3f;
    t[word::scene_n] = 400; std::memcpy(&t[word::scene_d], &d, sizeof(d));
    t[word::scene_state] = 1u | 2u | std::uint32_t(ui_detection::scene_verdict::visible) << 2; t[word::scene_decided] = 600;
    t[word::hudless_scene_n] = 300; std::memcpy(&t[word::hudless_scene_d], &hudless_d, sizeof(hudless_d));
    t[word::hudless_scene_state] = 3u;
    const auto sample = ui_temporal::decode_detection_sample(t.data(), t.size(), 1500, 9, true, true);
    const auto &e = sample.evidence;
    require(sample.source_kind == 2 && sample.enabled && sample.state == alpha_auto_state::automatic_on &&
        sample.covered == 995 && sample.pixels == 1000 && sample.sample_tick_ms == 1500 && sample.sample_sequence == 9 &&
        sample.accepted_samples == 9 && e.matching_tiles == 7 && e.candidates == 0x3a && e.hudless_changed == 9 &&
        e.hudless_unchanged == 900 && e.hudless_invalid == 1 && e.hudless_lit == 800 && e.trusted_alpha == 2 &&
        e.ui_layer && e.alpha_covered[3] == 13 && e.alpha_invalid[0] == 20 && e.alpha_opaque[1] == 31,
      "The decision texels did not decode");
    require(e.scene.n == 400 && e.scene.d == .5f && e.scene.valid && e.scene.ran &&
        e.scene.verdict == ui_detection::scene_verdict::visible && e.scene.decided == 600 && e.hudless_scene.n == 300 &&
        e.hudless_scene.d == .3f && e.hudless_scene.valid && e.hudless_scene.ran, "The scene texels did not decode");
    const auto unsupported = ui_temporal::decode_detection_sample(t.data(), t.size(), 1500, 9, false, false);
    require(!unsupported.evidence.scene.ran && !unsupported.evidence.scene.n && !unsupported.evidence.ui_layer,
      "Scene texels decoded without scene evidence");
    require(!ui_temporal::decode_detection_sample(t.data(), 4, 1500, 9, false, false).pixels,
      "Too few texels decoded");
    // 995 of 1000 pixels from alpha source 2 is a whole-frame alpha; its
    // measured verdict is counted apart from the full-frame routes'.
    namespace n = ui_counter;
    require(ui_temporal::full_alpha(sample), "A 99.5% alpha decision was not whole-frame");
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
  }
} // namespace

int main() {
  try {
    mode_is_auto_until_a_manual_edit_and_auto_again_after_it();
    alpha_trust_is_earned_by_selective_coverage_and_lost_on_contradiction();
    remembered_alpha_trust_is_restored_and_every_change_is_reported();
    trust_belongs_to_the_source_that_filled_the_slot();
    only_the_deciding_inputs_key_a_status_sample();
    hidden_scene_gates_and_samples_that_flatten_learn_nothing();
    temporal_state_holds_and_scene_verdicts();
    ui_counters_are_formatted_and_trust_events_counted();
    sample_decode_and_commit_are_shared();
    std::puts("Source alpha session: 9 policy groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Source alpha session failed: %s\n", error.what());
    return 1;
  }
}
