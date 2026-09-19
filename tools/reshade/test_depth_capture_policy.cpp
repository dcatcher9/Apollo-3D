// SPDX-License-Identifier: GPL-3.0-only
#include "depth_capture_policy.h"
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_depth_capture;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  source_observation source(std::uint64_t id, bool qualified = true) {
    return {id, 1, 10, {3840, 2160, 40, 0, 0, 3840, 2160}, qualified};
  }
  capture_record copy(std::uint64_t id) {
    capture_record result;
    result.source = 100 + id; result.identity = id; result.sampled = 200 + id;
    result.assignment = 300 + id; result.view = 400 + id;
    result.runtime = 3; result.present = 20; result.frame = 10; result.layout = 1;
    result.shape = source(id).shape; result.active = true; result.preserved = true;
    return result;
  }
  capture_set captures() {
    capture_set result;
    result.present = 20; result.frame = 10; result.runtime = 3; result.count = 3;
    for (unsigned i = 0; i != result.count; ++i) result.records[i] = copy(i + 1);
    return result;
  }
  preference preferred() { return {1, {1, 2, 3, 0}, 3, false}; }

  void allocation_is_not_selection() {
    budget slots;
    std::vector<source_observation> live {source(1, false), source(2, false), source(3)};
    auto set = slots.update(1, live, false);
    require(set.count == 2 && set.members[0] == 1 && set.members[1] == 3,
      "Discovery anchor must allocate before qualification; unrelated unqualified sources cannot take slots");
    auto frames = captures();
    auto ranking = preferred(); ranking.qualified = {3, 0, 0, 0}; ranking.count = 1;
    require(choose_current(frames, ranking).source == 3, "Unqualified anchor overruled a qualified current copy");
    ranking.count = 0;
    require(choose_current(frames, ranking).reason == selection_status::unqualified, "Unqualified current storage rendered automatically");
    ranking.manual = true;
    require(choose_current(frames, ranking).source == 1, "Explicit pin/statistics choice was made dependent on automatic qualification");
  }
  void overlapping_copies_keep_anchor() {
    auto frames = captures(); auto ranking = preferred();
    require(choose_current(frames, ranking).source == 1, "Concurrent independent copies rejected the stable anchor");
    frames.records[0].preserved = false;
    require(choose_current(frames, ranking).source == 2, "Missing anchor copy did not use qualified current peer");
    frames.records[1].ambiguous = true;
    require(choose_current(frames, ranking).source == 3, "Same-copy ambiguity did not reject only its own source");
    frames.records[2].active = false;
    require(!choose_current(frames, ranking).source, "Stale/inactive storage rendered");
    ranking.manual = true;
    require(choose_current(captures(), ranking).source == 1, "Exact manual source not chosen");
    require(choose_current(frames, ranking).reason == selection_status::manual_missing, "Manual pin silently switched resources");
  }
  void exact_currentness() {
    const auto good = copy(1);
    require(good.current(20, 10, 3) && good.renderable(20, 10, 3), "Valid copy rejected");
    for (unsigned kind = 0; kind != 16; ++kind) {
      auto value = good;
      switch (kind) {
        case 0: value.source = 0; break;
        case 1: value.identity = 0; break;
        case 2: value.sampled = 0; break;
        case 3: value.assignment = 0; break;
        case 4: value.runtime = 4; break;
        case 5: value.present = 19; break;
        case 6: value.frame = 9; break;
        case 7: value.layout = 0; break;
        case 8: value.active = false; break;
        case 9: value.ambiguous = true; break;
        case 10: value.preserved = false; break;
        case 11: value.capture_after = 10; break;
        case 12: value.shape.width = 0; break;
        case 13: value.shape.x = value.shape.width; break;
        case 14: value.shape.extent_height = value.shape.height + 1; break;
        case 15: value.shape.extent_width = 0; break;
      }
      require(!value.current(20, 10, 3), "Invalid exact-copy field was accepted");
    }
    auto challenger = good; challenger.view = 0;
    require(challenger.current(20, 10, 3) && !challenger.renderable(20, 10, 3), "Probe sampling depends on a renderer SRV");
    auto fresh = good; fresh.capture_after = 9;
    require(fresh.current(20, 10, 3), "First later legal assignment capture rejected");
    auto direct = good; direct.preserved = false; direct.direct = true;
    require(direct.current(20, 10, 3), "Legal direct-access reference source rejected");
    auto frames = captures(); frames.runtime = 4;
    require(!choose_current(frames, preferred()).source, "Different runtime records accepted by capture set");
    frames = captures(); frames.count = maximum_sources + 1;
    require(!choose_current(frames, preferred()).source, "Unbounded capture count read");
  }
  void retained_lifetimes_and_budget() {
    budget slots;
    std::vector<source_observation> live {source(1), source(2), source(3), source(4)};
    auto first = slots.update(1, live, false);
    require(first.count == 4, "Qualified budget not filled");
    for (auto &item : live) { item.last_used = 0; item.qualified = false; }
    auto idle = slots.update(1, live, false);
    require(idle.members == first.members && idle.epoch == first.epoch, "Temporary inactivity/flat content evicted live numeric ownership");
    live.push_back(source(5));
    auto replacement = slots.update(1, live, false);
    require(replacement.count == 4 && replacement.members[0] == 1 &&
      std::find(replacement.members.begin(), replacement.members.end(), 5) != replacement.members.end(), "New qualified source could not replace an unqualified peer within bounded budget");
    live.erase(std::remove_if(live.begin(), live.end(), [](const auto &o) { return o.id == 3; }), live.end());
    auto destroyed = slots.update(1, live, false);
    require(std::find(destroyed.members.begin(), destroyed.members.begin() + destroyed.count, 3) == destroyed.members.begin() + destroyed.count,
      "Destroyed source survived authoritative roster");
    for (auto &item : live) if (item.id == 5) { ++item.layout; item.shape.x = 1; --item.shape.extent_width; }
    auto recrop = slots.update(1, live, false);
    require(std::find(recrop.members.begin(), recrop.members.begin() + recrop.count, 5) == recrop.members.begin() + recrop.count,
      "Changed captured crop inherited compatible slot");
    auto pinned = slots.update(1, live, true);
    require(pinned.count == 1 && pinned.members[0] == 1 && pinned.epoch == first.epoch, "Same-source pin restarted capture generation");
    require(slots.update(1, live, false).epoch == pinned.epoch, "Same-source unpin restarted capture generation");
    live.push_back(source(100));
    auto changed = slots.update(100, live, false);
    require(changed.epoch > first.epoch && changed.members[0] == 100, "New anchor outside budget did not replace generation");
  }
}
int main() {
  try {
    allocation_is_not_selection(); overlapping_copies_keep_anchor(); exact_currentness(); retained_lifetimes_and_budget();
    std::puts("depth_capture_policy: 4 cases passed"); return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
