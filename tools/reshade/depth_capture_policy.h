// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace sunshine_depth_capture {
  inline constexpr unsigned maximum_sources = 4;

  struct geometry {
    std::uint32_t width{}, height{}, format{}, x{}, y{}, extent_width{}, extent_height{};
    bool operator==(const geometry &b) const {
      return width == b.width && height == b.height && format == b.format && x == b.x && y == b.y &&
        extent_width == b.extent_width && extent_height == b.extent_height;
    }
  };

  // Slow source evidence controls only capture allocation/preferences. It does
  // not assert a current copy, a shared numeric basis, or exclusive rendering.
  struct source_observation {
    std::uint64_t id{}, layout{}, last_used{};
    geometry shape;
    bool qualified{};
  };
  struct source_set {
    std::uint64_t epoch{};
    std::array<std::uint64_t, maximum_sources> members{};
    unsigned count{};
  };

  class budget {
    source_set sources_;
    std::array<source_observation, maximum_sources> retained_{};
  public:
    void reset() {
      const auto epoch = sources_.epoch + 1;
      *this = {};
      sources_.epoch = epoch;
    }
    source_set update(std::uint64_t anchor, const std::vector<source_observation> &live, bool manual) {
      const auto find = [&](std::uint64_t id) {
        return std::find_if(live.begin(), live.end(), [id](const auto &o) { return id && o.id == id; });
      };
      const auto primary = find(anchor);
      if (primary == live.end()) {
        if (sources_.count) reset();
        if (!sources_.epoch) sources_.epoch = 1;
        return sources_;
      }
      source_set next;
      next.epoch = sources_.epoch ? sources_.epoch : 1;
      std::array<source_observation, maximum_sources> kept{};
      const auto append = [&](const source_observation &o) {
        next.members[next.count] = o.id;
        kept[next.count++] = o;
      };
      append(*primary);
      if (!manual) {
        // A temporarily inactive/flat member still owns the same capture slot
        // and numeric basis. Remove it only on authoritative tuple change or
        // replacement by a newly qualified source when the budget is full.
        for (unsigned i = 0; i != sources_.count && next.count < maximum_sources; ++i) {
          const auto old = find(sources_.members[i]);
          if (old == live.end() || old->id == anchor || !(old->shape == primary->shape) ||
              old->layout != retained_[i].layout || !(old->shape == retained_[i].shape)) continue;
          append(*old);
        }
        std::vector<const source_observation *> candidates;
        for (const auto &o : live)
          if (o.id && o.qualified && o.shape == primary->shape &&
              std::find(next.members.begin(), next.members.begin() + next.count, o.id) == next.members.begin() + next.count)
            candidates.push_back(&o);
        std::sort(candidates.begin(), candidates.end(), [](const auto *a, const auto *b) { return a->id < b->id; });
        for (const auto *o : candidates) {
          if (next.count < maximum_sources) append(*o);
          else {
            unsigned replace = maximum_sources;
            for (unsigned i = 1; i != next.count; ++i)
              if (!kept[i].qualified && (replace == maximum_sources || kept[i].last_used < kept[replace].last_used)) replace = i;
            if (replace == maximum_sources) break;
            next.members[replace] = o->id; kept[replace] = *o;
          }
        }
      }
      // Physical rotation never changes this generation. A real preferred
      // source outside the retained set starts a new capture budget identity.
      if (sources_.count && std::find(sources_.members.begin(), sources_.members.begin() + sources_.count, anchor) ==
          sources_.members.begin() + sources_.count) ++next.epoch;
      sources_ = next; retained_ = kept;
      return sources_;
    }
  };

  // Immutable result of capture ownership for one native present. All handles
  // are borrowed callback-scoped identities; this value performs no GPU work.
  struct capture_record {
    std::uint64_t source{}, identity{}, sampled{}, assignment{}, view{};
    std::uint64_t runtime{}, present{}, frame{}, layout{}, capture_after{};
    geometry shape;
    bool active{}, preserved{}, ambiguous{}, direct{};

    bool current(std::uint64_t expected_present, std::uint64_t expected_frame, std::uint64_t expected_runtime) const {
      return source && identity && sampled && assignment && runtime && present && frame && layout &&
        runtime == expected_runtime && present == expected_present && frame == expected_frame && active && !ambiguous &&
        shape.width && shape.height && shape.extent_width && shape.extent_height && shape.x < shape.width && shape.y < shape.height &&
        shape.extent_width <= shape.width - shape.x && shape.extent_height <= shape.height - shape.y &&
        (preserved || direct) && (!capture_after || frame > capture_after);
    }
    bool renderable(std::uint64_t expected_present, std::uint64_t expected_frame, std::uint64_t expected_runtime) const {
      return view && current(expected_present, expected_frame, expected_runtime);
    }
  };
  struct capture_set {
    std::uint64_t present{}, frame{}, runtime{};
    std::array<capture_record, maximum_sources> records{};
    unsigned count{};
  };
  struct preference {
    std::uint64_t anchor{};
    std::array<std::uint64_t, maximum_sources> qualified{};
    unsigned count{};
    bool manual{};
  };
  enum class selection_status { current_anchor, current_peer, no_current_copy, unqualified, manual_missing };
  struct selection_result { std::uint64_t source{}; selection_status reason{selection_status::no_current_copy}; };
  inline const char *name(selection_status value) {
    switch (value) {
      case selection_status::current_anchor: return "current_anchor";
      case selection_status::current_peer: return "current_peer";
      case selection_status::no_current_copy: return "no_current_copy";
      case selection_status::unqualified: return "unqualified";
      case selection_status::manual_missing: return "manual_missing";
    }
    return "unknown";
  }
  inline selection_result choose_current(const capture_set &captures, const preference &preferred) {
    if (captures.count > maximum_sources || preferred.count > maximum_sources) return {};
    const auto current = [&](std::uint64_t id) -> const capture_record * {
      for (unsigned i = 0; i != captures.count; ++i)
        if (captures.records[i].identity == id && captures.records[i].renderable(captures.present, captures.frame, captures.runtime)) return &captures.records[i];
      return nullptr;
    };
    const auto qualified = [&](std::uint64_t id) {
      return std::find(preferred.qualified.begin(), preferred.qualified.begin() + preferred.count, id) !=
        preferred.qualified.begin() + preferred.count;
    };
    if (current(preferred.anchor) && (preferred.manual || qualified(preferred.anchor)))
      return {preferred.anchor, selection_status::current_anchor};
    if (preferred.manual) return {0, selection_status::manual_missing};
    for (unsigned i = 0; i != preferred.count; ++i)
      if (current(preferred.qualified[i])) return {preferred.qualified[i], selection_status::current_peer};
    for (unsigned i = 0; i != captures.count; ++i)
      if (captures.records[i].renderable(captures.present, captures.frame, captures.runtime)) return {0, selection_status::unqualified};
    return {};
  }
}
