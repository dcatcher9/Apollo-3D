// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The one reading of DLSS-G's generated count. The owning contract is
// docs/reshade-sbs.md, Presentation scope.

#include <cstdint>

namespace sunshine_streamline::frame_generation {
  // The Presents DLSS-G adds per real frame: the generated count the game
  // declared, 1 while FG is enabled without one (Dynamic mode with
  // numFramesToGenerate 0, whose count DLSS-G may pick), 0 without FG. Every
  // reader of a count takes it so: a viewport's FG options
  // (frame_generation_snapshot), the Present's grant (present_fg) and the
  // UI's Present counting (ui_evidence). Until 11-01 the deferral clock and
  // DLSS-G's evaluation window took Dynamic's 0 as no generated Present (F4),
  // and until the second fix review the rule was written out at each reader.
  constexpr std::uint32_t presents_per_frame(bool enabled, std::uint32_t generated_frames) {
    return enabled ? (generated_frames ? generated_frames : 1u) : 0u;
  }
  static_assert(presents_per_frame(true, 3) == 3 && presents_per_frame(true, 0) == 1 &&
    presents_per_frame(false, 3) == 0 && presents_per_frame(false, 0) == 0);
}
