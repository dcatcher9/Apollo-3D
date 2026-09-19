// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>

namespace sunshine_depth_probe {
  // Waiting for safe allocation and collecting captures are separate bounded
  // phases. The caller starts this clock once on queueing and once on allocation.
  inline bool phase_expired(std::uint64_t started_ms, std::uint64_t now_ms) {
    return now_ms < started_ms || now_ms - started_ms >= 2000;
  }

  // A completed raw measurement owns only its own visit. In particular an
  // unrelated completion cannot cancel a chosen probe waiting for retirement
  // to permit backup allocation. Allocation and the visit deadline stay with
  // the caller; this decision grants neither capture nor content qualification.
  inline bool keep_unrelated_probe(bool raw_completion, std::uint64_t completed,
                                   std::uint64_t pending) {
    return raw_completion && pending && completed != pending;
  }

  struct capture_owner {
    std::uint64_t source{}, view{}, retire_after{};
  };
  template <std::size_t Count>
  bool capture_keeps_backup(std::uint64_t source, const std::array<capture_owner, Count> &owners) {
    for (const auto &owner : owners)
      if (source && owner.source == source && owner.view && !owner.retire_after) return true;
    return false;
  }

  class discovery_hint {
    struct candidate {
      std::uint64_t id{}, layout{}, seen_frame{};
      unsigned observations{};
    };
    std::array<candidate, 4> candidates_{};
    std::uint64_t anchor_{}, frame_{};
    bool yield_to_round_robin_{};
  public:
    // candidate is the strongest eligible, same-geometry raster source on an
    // uncovered present. Zero covers ordinary/off-turn frames. Repeated effect
    // callbacks cannot count twice. Up to three intervening presents permit
    // four-way rotation without retaining stale hints. Each candidate keeps its
    // own observation: uncovered B/C/B/C must not reset each other's progress.
    void observe(std::uint64_t anchor, std::uint64_t frame,
                 std::uint64_t candidate, std::uint64_t layout) {
      if (!anchor || anchor != anchor_ || frame < frame_) {
        *this = {};
        anchor_ = anchor;
      }
      if (!anchor || !frame || frame == frame_) return;
      frame_ = frame;
      for (auto &entry : candidates_)
        if (entry.id && frame - entry.seen_frame > 4) entry = {};
      if (!candidate || !layout) return;
      for (auto &entry : candidates_) {
        if (entry.id != candidate) continue;
        entry.observations = entry.layout == layout ? 2 : 1;
        entry.layout = layout;
        entry.seen_frame = frame;
        return;
      }
      auto *slot = &candidates_.front();
      for (auto &entry : candidates_)
        if (!entry.id) { slot = &entry; break; }
        else if (entry.seen_frame < slot->seen_frame) slot = &entry;
      *slot = {candidate, layout, frame, 1};
    }

    std::uint64_t take(std::uint64_t frame) {
      // Queue ownership from recent observed raster activity. Requiring the
      // peer to render on the sampler's exact free frame can phase-lock against
      // rotation. The caller still requires an actual current copy to sample or
      // render; this hint does not make any older pixels current.
      if (yield_to_round_robin_) return 0;
      const candidate *latest = nullptr;
      for (const auto &entry : candidates_)
        if (entry.observations >= 2 && frame >= entry.seen_frame && frame - entry.seen_frame <= 4 &&
            (!latest || entry.seen_frame > latest->seen_frame)) latest = &entry;
      if (!latest) return 0;
      yield_to_round_robin_ = true;
      return latest->id;
    }
    // Even an uncapturable recurring peer gets at most every other visit.
    // Refreshing the incumbent does not count as discovering another source.
    void round_robin_started() { yield_to_round_robin_ = false; }
  };
}
