// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_hold.h"

namespace sunshine_streamline::depth_capture {
  // FG keeps its viewport while it can supply. An enabled FG scope that is
  // silent at this Present (its newest frame had no FG copy) lets the same
  // viewport's ordinary SR/RR depth stand in only once its own newest copy
  // can no longer be held: while the display holds a copy of that scope made
  // since FG On (policy.enabled_tick), younger than the source age, that copy
  // is held over a fresh or pending ordinary candidate. A frame without an FG
  // copy is then held as any missing frame of a source is, instead of
  // switching the viewport's source (which restarts the placement
  // calibration) to SR and back. policy: the confirmed FG policy, not the
  // silent scope's selection.
  inline bool frame_generation_holds(const sunshine_game3d::depth_output::retained_depth &previous, selection_policy policy, std::uint64_t now) {
    const auto &before = previous.metadata;
    return policy.require_frame_generation && policy.epoch && sunshine_game3d::depth_output::frame_generation_copy(previous, now) &&
      before.epoch == policy.epoch && before.viewport == policy.viewport && before.tick >= policy.enabled_tick;
  }

  // FG Off retires the FG source, which releases its queue's API association
  // (retire_source), while the display may still hold that scope's newest
  // copy. Within the source age it is held as any source's missing frame is:
  // while the display holds an FG copy younger than the source age and FG is
  // not revoked (an unconfirmed scope refuses its copies), the queue stays
  // the API's, so decide_display holds the copy until it expires, a readable
  // frame of another source replaces it or a source outside its lineage
  // drops it. Only then may the queue hand over to Generic. The successor's
  // copies recorded since the shown FG copy stay admissible at FG Off
  // (retire_source retires only the FG role), so the first of them readable
  // replaces the held copy at once.
  inline bool frame_generation_retained(const sunshine_game3d::depth_output::retained_depth &previous, selection_policy policy, std::uint64_t now) {
    return !policy.exclude_unconfirmed_fg && sunshine_game3d::depth_output::frame_generation_copy(previous, now);
  }
}
