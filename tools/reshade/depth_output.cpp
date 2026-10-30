// SPDX-License-Identifier: GPL-3.0-only
#include "depth_output.h"

namespace sunshine_game3d::depth_output {
  namespace {
    // Marks a Generic capture's scene identity, which shares no numbering with
    // the providers' epochs and sequences. Constant-initialized, with no state.
    constexpr std::uint64_t generic_scene_source = std::uint64_t{1} << 63;
    bool generic_depth(const sunshine_depth::frame_depth &depth) {
      return depth.ready && !depth.provided.epoch && !depth.provided.sequence;
    }
  }

  shown_depth describe(const sunshine_depth::frame_depth &depth) {
    shown_depth shown;
    shown.generic = generic_depth(depth);
    shown.generic_ui = (!depth.provided.sequence && depth.ready && !depth.reused_depth) ||
      (shown.generic && depth.reused_depth);
    shown.ready = depth.ready;
    shown.scene_source = shown.generic ?
      std::array<std::uint64_t, 4>{depth.runtime_epoch, depth.source_id, depth.frame_index, depth.layout_epoch | generic_scene_source} :
      std::array<std::uint64_t, 4>{depth.provided.epoch, depth.provided.source_id, depth.provided.sequence, depth.provided.viewport};
    auto &input = shown.ui_source;
    input.tick_ms = depth.provided.tick;
    input.epoch = depth.provided.epoch;
    input.revision = depth.provided.observation_revision;
    input.source_id = depth.provided.source_id;
    input.sequence = depth.provided.sequence;
    input.viewport = depth.provided.viewport;
    if (shown.generic_ui) {
      input.epoch = depth.runtime_epoch;
      input.revision = depth.layout_epoch;
      input.source_id = depth.source_id;
      input.sequence = depth.frame_index;
    }
    auto &current = shown.raw_source;
    current.layout_epoch = depth.layout_epoch;
    current.source = {depth.source_resource.handle, depth.source_id, 0, depth.width, depth.height,
      depth.x, depth.y, depth.active_width, depth.active_height};
    current.frame = {depth.frame_index, depth.runtime_epoch};
    current.direction = depth.detected_orientation;
    current.depth_ready = depth.ready;
    current.aligned_viewport_assumed = depth.aligned_viewport_assumed;
    return shown;
  }

  bool same_capture(const source_identity &source, const source_identity &previous) {
    return source[0] && source[2] && previous == source;
  }

  bool same_source_except_sequence(const source_identity &previous, const source_identity &source) {
    return !(previous[0] != source[0] || previous[1] != source[1] || previous[3] != source[3]);
  }

  std::uint64_t depth_identity(const source_identity &source, bool borrowed_depth, bool depth_ready) {
    if (!borrowed_depth || !depth_ready || !source[0] || !source[2] ||
        (source[3] & generic_scene_source)) return 0;
    std::uint64_t value = 0x9e3779b97f4a7c15ull;
    for (const auto part : source) {
      value ^= part + 0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2);
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
      value ^= value >> 31;
    }
    return value ? value : 1;
  }

  ui_adaptive::source ui_observation(const shown_depth &shown, const ui_adaptive::source &previous,
      bool scene_ready, std::uint64_t generic_routing_epoch, std::uint64_t raw_basis_epoch, std::uint64_t now) {
    auto input = shown.ui_source;
    input.now_ms = now;
    input.eligible = shown.ready && scene_ready;
    if (shown.generic_ui) {
      // A held Generic capture is the same capture, so it keeps its tick.
      // Generic capture identity remains physical provenance. Placement follows
      // the admitted capture group across its rotating, independently calibrated
      // members; per-resource layout epochs cannot identify that logical group.
      input.generic_basis_epoch = raw_basis_epoch;
      input.generic_routing_epoch = generic_routing_epoch;
      input.eligible = input.eligible && input.generic_basis_epoch && input.generic_routing_epoch;
      input.tick_ms = previous.epoch == input.epoch && previous.revision == input.revision &&
        previous.source_id == input.source_id && previous.sequence == input.sequence ? previous.tick_ms : now;
    }
    return input;
  }

  sunshine_raw_scene::selected_frame raw_frame(const shown_depth &shown, std::uint64_t basis_epoch) {
    auto current = shown.raw_source;
    current.basis_epoch = basis_epoch;
    return current;
  }
}
