// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Pure UI capture/presentation evidence. The owning contract is
// docs/reshade-sbs.md, UI source discovery and snapshot qualification.
#include "frame_generation_count.h"
#include "game3d_ui_qualification.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace sunshine_game3d::ui_evidence {
  // The presented color a HUD-less image is compared with when no same-batch
  // Backbuffer was tagged with it. Without frame generation every offer of a
  // snapshot counts from its tag: the Present after the tag (next_present,
  // presents_ago 0), or, for a capture that completed later (another queue)
  // and for a re-offer, the color retained up to max_late_presents Presents
  // ago (earlier_present), the snapshot's own frame; anything older is
  // unpaired. Under frame generation the Present after the tag may be
  // generated (DLSS-G presents its generated frames first) and the real frame
  // come later in the snapshot's window, so only a snapshot's first offer on
  // that Present counts; a later first offer and every later offer of the
  // same snapshot (the capture owner offers the newest ready snapshot of its
  // kind for as long as it is recent) pair with the current Present's own
  // color (reoffered, presents_ago 0), which may be the real frame's. This
  // assumes DLSS-G's generated-first order: a real frame presented first is
  // compared only on the Present after the tag.
  // Counting Presents only proposes these pairs: a game may tag its next frame
  // before the Present counted for it, and frame generation presents
  // interpolated color between real frames, so the pair is never exact (UI
  // framework E2). Only V2's test of the pixels validates it, and only a
  // same-batch Backbuffer (same_tag_interval) makes a pair exact.
  enum class hudless_present { unpaired, next_present, earlier_present, reoffered };
  inline constexpr std::uint32_t max_late_presents = 2;
  struct hudless_pairing {
    hudless_present kind = hudless_present::unpaired;
    std::uint32_t presents_ago = 0; // Nonzero only for earlier_present.
  };
  constexpr hudless_pairing pair_hudless_present(std::uint64_t tagged, std::uint64_t current) {
    if (!tagged || tagged == UINT64_MAX || current <= tagged) return {};
    const auto ago = current - tagged - 1;
    if (!ago) return {hudless_present::next_present};
    if (ago > max_late_presents) return {};
    return {hudless_present::earlier_present, static_cast<std::uint32_t>(ago)};
  }
  // One offer of a snapshot: offered_before when the provider already offered
  // this snapshot (its ticket, epoch and viewport) on an earlier Present;
  // fg_active while frame generation is active.
  constexpr hudless_pairing pair_hudless_offer(std::uint64_t tagged, std::uint64_t current, bool offered_before,
      bool fg_active) {
    const auto counted = pair_hudless_present(tagged, current);
    if (!fg_active || (!offered_before && counted.kind == hudless_present::next_present)) return counted;
    if (!tagged || tagged == UINT64_MAX || current <= tagged) return {};
    return {hudless_present::reoffered};
  }
  // Present counting of generated Presents (UI framework T1). Under frame
  // generation a game may tag its UI inputs (UIAlpha, the UI color tag, the
  // Backbuffer) for real frames only, and a generated Present may then offer
  // none of them: Expedition 33's Backbuffer at 2x, 3x or 4x. Such a Present
  // within the reported generated count (one while the count is unknown) of
  // the last Present that offered one shows that Present's real frame, so it
  // is generated and shows that decision. Presents are numbered by the
  // caller, one per render; zero is none. A Present that offers any input (a
  // HUD-less image or the offscreen UI layer included), or one without frame
  // generation or beyond the count, is real, and when it misses an accepted
  // input the T1 grace reuses the previous real decision once. No multiplier
  // constant: a count reported too high holds only Presents without input of
  // their own, and one too low leaves the later ones to the grace.
  constexpr bool generated_without_input(std::uint64_t present, std::uint64_t input_present, bool fg_active,
      std::uint32_t generated_frames) {
    if (!fg_active || !input_present || present <= input_present) return false;
    return present - input_present <= sunshine_streamline::frame_generation::presents_per_frame(true, generated_frames);
  }
  static_assert(generated_without_input(11, 10, true, 3) && generated_without_input(13, 10, true, 3) &&
    !generated_without_input(14, 10, true, 3) && generated_without_input(11, 10, true, 0) &&
    !generated_without_input(12, 10, true, 0) && !generated_without_input(11, 10, false, 3) &&
    !generated_without_input(11, 0, true, 3) && !generated_without_input(10, 10, true, 3));
  // T1 by depth's window (PD7, 11-01). Where depth named the frame the last
  // Present that offered a UI tag presented (input_frame: the presenting
  // frame of that Present's API depth acquisition, present_fg::depth_frame),
  // a later Present without input of its own is that frame's generated
  // Present exactly while depth still presents that frame there (frame ==
  // input_frame): depth's window, which the next frame's first evaluation
  // ends and the Present's grant bounds, so UI and depth take one decision.
  // Until 11-01 the UI counted its own window of the generated count from
  // that Present, so where the next frame's evaluation began inside it
  // (DLSS-G presenting earlier frames inside the next frame's recording;
  // multi-frame generation) the next frame's Presents held the earlier
  // frame's decision, or the reverse after a freeze. Without a depth window
  // at the input (input_frame 0: no API depth, or an API scope without an
  // open window at that Present) the count decides (generated_without_input).
  constexpr bool generated_in_window(std::uint64_t present, std::uint64_t input_present, bool fg_active,
      std::uint32_t generated_frames, std::uint64_t frame, std::uint64_t input_frame) {
    if (!input_frame) return generated_without_input(present, input_present, fg_active, generated_frames);
    return fg_active && input_present && present > input_present && frame == input_frame;
  }
  static_assert(generated_in_window(11, 10, true, 1, 7, 7) && generated_in_window(15, 10, true, 1, 7, 7) &&
    !generated_in_window(11, 10, true, 3, 8, 7) && !generated_in_window(11, 10, true, 3, 0, 7) &&
    !generated_in_window(11, 10, false, 3, 7, 7) && !generated_in_window(10, 10, true, 3, 7, 7) &&
    generated_in_window(13, 10, true, 3, 8, 0) && !generated_in_window(14, 10, true, 3, 8, 0));
  // T1 by snapshot identity. The capture owner offers each kind's newest ready
  // snapshot on every Present while it is recent, and the layer its newest
  // copy, so under frame generation a generated Present re-offers the real
  // frame's inputs instead of offering none. A Present that offers exactly the
  // snapshot identities the last detecting Present offered (the UIAlpha, UI
  // color and Backbuffer tickets, an exact HUD-less pair's ticket and the
  // layer copy's id, at least one of them) shows that real frame, and holds
  // its decision as Present counting does. Current colour and an inexact
  // HUD-less image (own_input) are the Present's own and never match. Without
  // frame generation every Present is real.
  inline constexpr std::size_t snapshot_input_slots = 5;
  using snapshot_inputs = std::array<std::uint64_t, snapshot_input_slots>;
  constexpr bool same_snapshot_inputs(const snapshot_inputs &offered, const snapshot_inputs &detected, bool fg_active,
      bool own_input) {
    if (!fg_active || own_input) return false;
    bool any = false;
    for (std::size_t i = 0; i != offered.size(); ++i) {
      if (offered[i] != detected[i]) return false;
      any = any || offered[i] != 0;
    }
    return any;
  }
  static_assert(same_snapshot_inputs({0, 0, 7, 0, 9}, {0, 0, 7, 0, 9}, true, false) &&
    !same_snapshot_inputs({0, 0, 7, 0, 9}, {0, 0, 7, 0, 9}, false, false) &&
    !same_snapshot_inputs({0, 0, 7, 0, 9}, {0, 0, 7, 0, 9}, true, true) &&
    !same_snapshot_inputs({0, 0, 8, 0, 9}, {0, 0, 7, 0, 9}, true, false) &&
    !same_snapshot_inputs({0, 0, 7, 0, 0}, {0, 0, 7, 0, 9}, true, false) &&
    !same_snapshot_inputs({}, {}, true, false));
  // Two captures belong to one tag batch (one game frame) when both were made
  // in the same Present interval. Each source keeps its own Present counter, so
  // compare Presents since each tag rather than the raw counter values.
  constexpr bool same_tag_interval(std::uint64_t tagged_a, std::uint64_t current_a,
      std::uint64_t tagged_b, std::uint64_t current_b) {
    const auto valid = [](std::uint64_t tagged, std::uint64_t current) {
      return tagged && tagged != UINT64_MAX && current >= tagged;
    };
    return valid(tagged_a, current_a) && valid(tagged_b, current_b) && current_a - tagged_a == current_b - tagged_b;
  }

  // The adapter supplies its pure format predicate; its source/status types
  // and the capture owner's types stay with their existing owners.
  template<class Status, class HasAlpha>
  const char *current_color_refusal(const Status &status, std::uint32_t output_format,
      ui_qualification::choice wanted, HasAlpha present_has_alpha) {
    using ui_qualification::choice;
    if (!status.requested) return "not_requested";
    if (status.fg_active()) return "FG_active";
    if (!present_has_alpha(output_format)) return "no_present_alpha";
    return wanted == choice::automatic || wanted == choice::current_color ? nullptr : "not_selected";
  }
}
