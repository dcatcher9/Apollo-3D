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
    ranking.single_anchor = true;
    require(choose_current(frames, ranking).source == 1, "Single-anchor statistics choice was made dependent on automatic qualification");
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
    ranking.single_anchor = true;
    require(choose_current(captures(), ranking).source == 1, "Exact single-anchor source not chosen");
    require(choose_current(frames, ranking).reason == selection_status::anchor_missing, "Single-anchor choice silently switched resources");
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
    auto single = slots.update(1, live, true);
    require(single.count == 1 && single.members[0] == 1 && single.epoch == first.epoch, "Same-source single anchor restarted capture generation");
    require(slots.update(1, live, false).epoch == single.epoch, "Same-source return from a single anchor restarted capture generation");
    live.push_back(source(100));
    auto changed = slots.update(100, live, false);
    require(changed.epoch > first.epoch && changed.members[0] == 100, "New anchor outside budget did not replace generation");
  }
  // An idle source (no work within the activity grace: a set the game stopped
  // drawing or abandoned) joins a capture set only in a slot no working source
  // takes, and a full set gives an idle member's slot to a working newly
  // qualified source; a member that only pauses keeps its slot while nothing
  // needs it.
  void idle_sources_and_budget() {
    const auto idle = [](std::uint64_t id) { auto o = source(id); o.idle = true; return o; };
    budget fresh;
    const std::vector<source_observation> crowded {idle(1), idle(2), idle(3), source(10), source(11)};
    auto set = fresh.update(10, crowded, false);
    require(set.count == 4 && set.members[0] == 10 && set.members[1] == 11 && set.members[2] == 1 && set.members[3] == 2,
      "Idle same-geometry peers with older lifetimes filled a new capture set and kept its working peer out");
    budget resume;
    const std::vector<source_observation> resuming {source(10), idle(11), idle(12)};
    set = resume.update(10, resuming, false);
    require(set.count == 3 && set.members[1] == 11 && set.members[2] == 12,
      "Idle peers lost the spare slots that let a resuming set's first turns be current");
    budget slots;
    std::vector<source_observation> live {source(1), source(2), source(3), source(4)};
    const auto first = slots.update(1, live, false);
    for (auto &item : live) if (item.id != 1) item.idle = true;
    const auto paused = slots.update(1, live, false);
    require(paused.members == first.members && paused.count == 4, "A pausing member lost its slot while no working source needed it");
    live.push_back(source(5));
    const auto replaced = slots.update(1, live, false);
    require(replaced.count == 4 && replaced.members[0] == 1 &&
      std::find(replaced.members.begin(), replaced.members.end(), 5) != replaced.members.end(),
      "A working qualified source could not take an idle member's slot in a full capture set");
    live.push_back(idle(6));
    const auto later = slots.update(1, live, false);
    require(std::find(later.members.begin(), later.members.begin() + later.count, 6) == later.members.begin() + later.count,
      "An idle source took a capture slot");
  }
  // Present coverage judges a candidate by the capture set it would route as
  // the anchor: alternating same-geometry qualified allocations count
  // together; another geometry or an unqualified allocation does not.
  void prospective_sets() {
    auto other = source(9); other.shape.width = other.shape.extent_width = 1920; other.shape.height = other.shape.extent_height = 1080;
    const std::vector<source_observation> pair {source(1), source(2)}, triple {source(1), source(2), source(3), other},
      unqualified {source(1), source(2, false)};
    require(prospective_set_contains(1, pair, {1}) && prospective_set_contains(1, pair, {2}) && prospective_set_contains(2, pair, {1}),
      "An A/B capture set was judged one member at a time");
    for (std::uint64_t id : {1u, 2u, 3u})
      require(prospective_set_contains(id, triple, {3}) && prospective_set_contains(id, triple, {1}),
        "An A/B/C capture set was judged one member at a time");
    require(!prospective_set_contains(9, triple, {1, 2, 3}) && !prospective_set_contains(1, triple, {9}),
      "Another geometry joined a prospective capture set");
    require(!prospective_set_contains(1, unqualified, {2}) && prospective_set_contains(2, unqualified, {1}),
      "An unqualified allocation joined another candidate's prospective set, or lost its own anchor's");
    require(!prospective_set_contains(1, pair, {}) && prospective_set_contains(4, pair, {4}),
      "A set without members among the ids drew, or a candidate outside the observations lost its own presence");
    // A new A/B set beside three idle members of an abandoned A/B/C set of the
    // same geometry (older lifetimes): judged as one set, not one member at a time.
    auto idle = [](std::uint64_t id) { auto o = source(id); o.idle = true; return o; };
    const std::vector<source_observation> second {idle(1), idle(2), idle(3), source(10), source(11)};
    require(prospective_set_contains(10, second, {11}) && prospective_set_contains(11, second, {10}),
      "Idle members of an abandoned set kept a new A/B set from being judged as one capture set");
    const std::vector<source_observation> full {idle(1), source(10), source(11), source(12), source(13)};
    require(!prospective_set_contains(10, full, {1}), "An idle source took a slot a working source needed");
    // Present coverage asks about two lists at once (worked, drew) from one set
    // build: each answer is the one-list answer for that list.
    const std::vector<std::vector<source_observation>> all {pair, triple, unqualified, second, full};
    const std::vector<std::vector<std::uint64_t>> lists {{}, {1}, {2}, {3}, {9}, {4}, {1, 2, 3}, {10}, {11}, {1, 11}, {13}};
    for (const auto &observations : all)
      for (std::uint64_t id : {1u, 2u, 3u, 4u, 9u, 10u, 11u, 13u})
        for (const auto &first : lists)
          for (const auto &more : lists) {
            const auto both = prospective_set_contains(id, observations, first, more);
            require(both.first == prospective_set_contains(id, observations, first) &&
              both.second == prospective_set_contains(id, observations, more),
              "One prospective set answered two id lists differently from one list at a time");
          }
  }
}
int main() {
  try {
    allocation_is_not_selection(); overlapping_copies_keep_anchor(); exact_currentness(); retained_lifetimes_and_budget();
    idle_sources_and_budget(); prospective_sets();
    std::puts("depth_capture_policy: 6 cases passed"); return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
