// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_adaptive.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace adaptive = sunshine_game3d::ui_adaptive;
namespace {
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  bool near(float a, float b) { return std::abs(double(a) - b) < 1.e-9; }
  adaptive::source source(std::uint64_t tick, float cap = .01f) {
    auto value = adaptive::source{tick, tick, 7, 4, 0, tick + 1, 0, 11, true};
    value.front_cap_uv = cap; return value;
  }
  adaptive::sample sample(std::uint64_t tick, unsigned width = 512, unsigned height = 512, float cap = .01f) {
    adaptive::sample value; value.provenance = source(tick, cap); value.width = width; value.height = height;
    for (unsigned y = 0; y != 16; ++y) for (unsigned x = 0; x != 16; ++x)
      value.tiles[y * 16 + x].pixels = ((x + 1) * width / 16 - x * width / 16) *
        ((y + 1) * height / 16 - y * height / 16);
    return value;
  }
  adaptive::sample counts(std::uint64_t tick, std::array<std::uint32_t, 5> bad, float cap = .01f) {
    auto value = sample(tick, 512, 512, cap); value.tiles[137].covered = 100; value.tiles[137].bad = bad; return value;
  }
  adaptive::sample required(std::uint64_t tick, unsigned level, float cap = .01f) {
    std::array<std::uint32_t, 5> bad {};
    for (unsigned i = 0; i < std::min(level, 5u); ++i) bad[i] = 100;
    return counts(tick, bad, cap);
  }
  void feed(adaptive::policy &policy, const adaptive::sample &value) {
    policy.prepare(value.provenance);
    require(policy.observe(value, value.provenance.now_ms), "Fresh well-formed source observation rejected");
  }
  void settle(adaptive::policy &policy, unsigned level, float cap = .01f) {
    for (const auto tick : {1000u, 1100u, 1200u, 1250u}) feed(policy, required(tick, level, cap));
    require(near(policy.current().applied_uv, adaptive::level_uv(std::min(level, 4u), cap)),
      "Confirmed approach did not settle at its absolute target");
  }

  void entry_boundary_and_absolute_levels() {
    adaptive::policy policy; feed(policy, counts(1000, {20, 0, 0, 0, 0}));
    require(policy.current().required_index == 0 && policy.current().target_uv == 0 && policy.current().conflict_pixels == 20,
      "Exactly twenty percent conflict moved the UI");
    feed(policy, counts(1100, {21, 0, 0, 0, 0}));
    require(policy.current().required_uv == .001f && policy.current().target_uv == 0, "Above twenty percent missed confirmation");
    feed(policy, counts(1200, {21, 0, 0, 0, 0}));
    require(policy.current().target_uv == .001f && policy.current().conflict_pixels == 0, "Confirmed entry chose the wrong level");
    for (unsigned index = 0; index < 5; ++index) {
      adaptive::policy level; settle(level, index);
      require(level.current().target_uv == adaptive::levels_uv[index] && level.current().target_index == index &&
          level.current().required_index == index, "An absolute level was interpreted as a scene-cap fraction");
    }
  }

  adaptive::sample aggregate_scene(std::uint64_t tick, std::uint32_t covered,
      const std::array<std::uint32_t, 5> &bad) {
    auto value = sample(tick, 640, 480); // Central rectangle: 480 * 360 = 172800 pixels.
    unsigned index{};
    for (unsigned y = 2; y < 14; ++y) for (unsigned x = 2; x < 14; ++x, ++index) {
      auto &cell = value.tiles[y * 16 + x];
      cell.covered = covered / 144 + (index < covered % 144);
      for (unsigned level = 0; level < 5; ++level)
        cell.bad[level] = bad[level] / 144 + (index < bad[level] % 144);
    }
    return value;
  }

  void area_rule_catches_diffuse_overlap_without_regions() {
    adaptive::policy policy;
    for (const auto tick : {1000u, 1100u, 1200u})
      feed(policy, aggregate_scene(tick, 46656, {6300, 5900, 3500, 3000, 0}));
    const auto &out = policy.current();
    require(out.center_pixels == 172800 && out.covered_pixels == 46656 &&
        out.conflict_counts == std::array<std::uint64_t, 5>{6300, 5900, 3500, 3000, 0} &&
        out.conflict_counts[0] * 100 < out.covered_pixels * adaptive::entry_conflict_percent &&
        out.target_uv == .003f && near(out.applied_uv, .003f) && out.required_index == 3,
      "Large safe UI diluted conflict occupying over two percent of the center");
    // Spatial distribution is immaterial: the accepted global counts decide.
    adaptive::policy small; auto value = sample(1000);
    value.tiles[2 * 16 + 2].covered = value.tiles[2 * 16 + 2].bad[0] = 100;
    value.tiles[2 * 16 + 5].covered = 900;
    feed(small, value);
    require(small.current().required_index == 0, "A small separated patch retained regional placement behavior");
    adaptive::policy tiny; value = sample(1000);
    value.tiles[2 * 16 + 2].covered = value.tiles[2 * 16 + 2].bad[0] = 1;
    feed(tiny, value); value.provenance = source(1100); feed(tiny, value);
    require(tiny.current().target_uv == .001f, "The area rule disabled the tiny-UI ratio trigger");
  }

  void exact_area_boundaries_and_both_release_tests() {
    adaptive::policy boundary;
    feed(boundary, aggregate_scene(1000, 17280, {3456, 3456, 3456, 3456, 3456}));
    require(boundary.current().required_index == 0 && !boundary.current().capped_conflict,
      "Exactly twenty percent UI and two percent center area triggered entry");
    feed(boundary, aggregate_scene(1100, 86400, {3457, 3457, 3457, 3457, 3457}));
    require(boundary.current().required_index == 4 && boundary.current().capped_conflict,
      "One pixel above two percent central area was ignored");

    adaptive::policy retreat; settle(retreat, 3);
    for (std::uint64_t tick = 1350; tick <= 3350; tick += 100)
      feed(retreat, aggregate_scene(tick, 86400, {2592, 2592, 2592, 0, 0}));
    require(retreat.current().target_uv == .003f,
      "Exactly 1.5 percent center area permitted retreat despite low UI conflict ratio");
    for (std::uint64_t tick = 3450; tick <= 5450; tick += 100)
      feed(retreat, aggregate_scene(tick, 10000, {1500, 1500, 1500, 0, 0}));
    require(retreat.current().target_uv == .003f,
      "Exactly fifteen percent UI conflict permitted retreat despite low center-area ratio");
    for (std::uint64_t tick = 5550; tick < 7050; tick += 100)
      feed(retreat, aggregate_scene(tick, 86400, {2591, 2591, 2591, 0, 0}));
    require(retreat.current().target_uv == .003f, "Both-clear retreat skipped its complete dwell");
    feed(retreat, aggregate_scene(7050, 86400, {2591, 2591, 2591, 0, 0}));
    require(retreat.current().target_uv == 0, "Both ratios below release thresholds did not permit retreat");
  }
  void sustained_approach_and_actual_slew() {
    adaptive::policy policy; feed(policy, required(1000, 4)); feed(policy, sample(1100));
    require(policy.current().target_uv == 0 && policy.current().applied_uv == 0, "Single-frame outlier moved the plane");
    feed(policy, required(1200, 1)); feed(policy, required(1250, 4));
    require(policy.current().target_uv == 0, "Two samples bypassed the 100ms confirmation interval");
    feed(policy, required(1300, 4));
    require(policy.current().target_uv == .001f, "One final spike replaced the sustained requirement");
    adaptive::policy fine, coarse;
    for (auto *p : {&fine, &coarse}) { feed(*p, required(1000, 5)); feed(*p, required(1100, 5)); }
    for (std::uint64_t tick = 1110; tick <= 1200; tick += 10) fine.prepare(source(tick));
    coarse.prepare(source(1200));
    require(near(fine.current().applied_uv, .003f) && near(fine.current().applied_uv, coarse.current().applied_uv),
      "Absolute approach rate depends on presentation frequency");
    fine.prepare(source(1250)); coarse.prepare(source(1250));
    require(near(fine.current().applied_uv, .0035f) && near(coarse.current().applied_uv, .0035f), "Approach missed the comfort cap");
  }

  void strict_release_and_conservative_dwell() {
    adaptive::policy policy; settle(policy, 4);
    for (std::uint64_t tick = 1350; tick <= 3350; tick += 100) feed(policy, counts(tick, {15, 15, 15, 15, 0}));
    require(policy.current().target_uv == .0035f, "Exactly fifteen percent conflict permitted retreat");
    for (std::uint64_t tick = 3450; tick < 4950; tick += 100) feed(policy, counts(tick, {14, 14, 14, 14, 0}));
    require(policy.current().target_uv == .0035f, "Retreat began before 1500ms clearance");
    feed(policy, counts(4950, {14, 14, 14, 14, 0}));
    require(policy.current().target_uv == 0 && near(policy.current().applied_uv, .0035f), "Clearance did not select zero or jumped the plane");
    feed(policy, counts(5050, {14, 14, 14, 14, 0}));
    require(near(policy.current().applied_uv, .003f), "Absolute retreat rate changed");
    adaptive::policy conservative; settle(conservative, 4);
    for (std::uint64_t tick = 1350; tick <= 2850; tick += 100)
      feed(conservative, required(tick, tick == 1450 ? 3 : 1));
    require(conservative.current().target_uv == .003f, "Retreat discarded the nearest safe candidate in its dwell");
    adaptive::policy clipped; settle(clipped, 5, .001f);
    for (std::uint64_t tick = 1350; tick <= 2850; tick += 100) feed(clipped, sample(tick, 512, 512, .001f));
    require(clipped.current().target_uv == 0, "Empty coverage trapped a clipped level above zero");
  }

  void caps_and_float_emission() {
    adaptive::policy policy; settle(policy, 5);
    require(policy.current().required_uv == .0035f && policy.current().required_index == 4 &&
        policy.current().limit_uv == .0035f && policy.current().capped_conflict && policy.current().conflict_pixels == 100,
      "Unresolved conflict escaped the absolute comfort cap");
    adaptive::policy relative; settle(relative, 5, .004f);
    require(relative.current().limit_uv == .002f && relative.current().applied_uv == .002f &&
        relative.current().applied_fraction == .5f && relative.current().target_index == 2 &&
        relative.current().capped_conflict && relative.current().conflict_pixels == 100, "Clipped duplicate levels broke the relative cap");
    adaptive::policy boundary;
    feed(boundary, counts(1000, {20, 20, 20, 20, 20}));
    require(!boundary.current().capped_conflict, "Exactly twenty percent capped conflict was mislabeled");
    feed(boundary, counts(1100, {21, 21, 21, 21, 21}));
    require(boundary.current().capped_conflict, "Above twenty percent capped conflict was hidden");
    for (const float cap : {.00814f, .007f, .0091f, .012f, .04f}) {
      adaptive::policy rounded; settle(rounded, 5, cap);
      const auto &out = rounded.current();
      require(out.applied_fraction <= .5f && out.applied_fraction * cap <= out.limit_uv &&
          out.applied_uv == out.applied_fraction * cap, "Serialized fraction rounded above the comfort cap");
    }
  }

  void cap_change_holds_actual_uv() {
    adaptive::policy policy; settle(policy, 2);
    const auto held = policy.current().applied_uv;
    policy.prepare(source(1300, .02f));
    require(near(policy.current().applied_uv, held) && near(policy.current().applied_fraction, held / .02f) &&
        policy.current().observed_front_cap_uv == .01f, "Nonbinding cap moved UI or relabeled old evidence");
    require(!policy.observe(required(1250, 5), 1300), "Old-cap observation was admitted");
    policy.prepare(source(1400, .02f));
    require(near(policy.current().applied_uv, held), "Cap change resumed an old ramp");
    feed(policy, required(1400, 2, .02f));
    require(policy.current().observed_front_cap_uv == .02f, "New accepted evidence did not record its actual display cap");
    policy.prepare(source(1450, .003f));
    require(policy.current().applied_uv == .0015f && policy.current().target_uv == .0015f &&
        policy.current().applied_fraction == .5f && policy.current().conflict_pixels == 100,
      "Shrinking cap failed to clamp displacement or reinterpreted old conflict thresholds");
    policy.prepare(source(1500, .02f));
    require(near(policy.current().applied_uv, .0015f), "Growing cap restored old displacement");
    require(!policy.observe(required(1400, 2, .02f), 1500), "Cap round-trip revived pre-change evidence");
    policy.prepare(source(1550, 0.f));
    require(policy.current().applied_uv == 0 && policy.current().applied_fraction == 0, "Zero cap retained displacement");
    auto invalid = source(1600); invalid.front_cap_uv = std::numeric_limits<float>::quiet_NaN(); policy.prepare(invalid);
    require(!policy.observe(required(1600, 2), 1600), "Invalid current cap admitted evidence");
  }

  void missing_duplicate_gap_and_clock() {
    adaptive::policy policy; settle(policy, 2);
    for (std::uint64_t tick = 1350; tick <= 2750; tick += 100) feed(policy, sample(tick));
    policy.prepare(source(2800)); require(!policy.observe(sample(2750), 2800), "Duplicate depth became fresh evidence");
    feed(policy, sample(2850)); require(policy.current().target_uv == .002f, "Duplicate completed release dwell");
    for (std::uint64_t tick = 2950; tick <= 4250; tick += 100) feed(policy, sample(tick));
    feed(policy, sample(4551)); require(policy.current().target_uv == .002f, "Gap completed release evidence");
    auto unavailable = source(4651); unavailable.eligible = false; const auto held = policy.current().applied_uv;
    policy.prepare(unavailable); policy.prepare(source(4700)); feed(policy, sample(4751));
    require(policy.current().applied_uv == held && policy.current().target_uv == .002f, "Missing input moved placement");
    adaptive::policy rise; feed(rise, required(1000, 5)); feed(rise, required(1300, 5));
    require(rise.current().target_uv == 0, "Gap completed approach confirmation");
    feed(rise, required(1400, 5)); rise.prepare(source(1390)); rise.prepare(source(1410));
    require(rise.current().applied_uv == 0, "Clock regression resumed motion");
  }

  void retained_mask_and_scope() {
    adaptive::policy policy; auto value = required(1000, 2); value.provenance.mask_sequence = 20; feed(policy, value);
    auto live = source(1100); live.mask_sequence = 20; policy.prepare(live); value.provenance = live;
    require(!policy.observe(value, 1100), "Repeated retained mask earned evidence");
    live.mask_sequence = 21; policy.prepare(live); value.provenance = live;
    require(policy.observe(value, 1100) && policy.current().target_uv == 0, "Duplicate mask kept confirmation evidence");
    live = source(1250); live.mask_sequence = 23; policy.prepare(live);
    value.provenance = source(1200); value.provenance.mask_sequence = 22;
    require(policy.observe(value, 1250) && policy.current().probe_age_ms == 50 && policy.current().target_uv == .002f,
      "Asynchronous source lost its original age or evidence interval");
    value.provenance = source(1260); value.provenance.mask_sequence = 24;
    require(!policy.observe(value, 1260), "Future unselected identity was admitted");
    live = source(1300); live.epoch = 99; live.eligible = false; policy.prepare(live);
    require(policy.current().accepted_sequence != 0, "Unavailable scope reset history");
    live.eligible = true; policy.prepare(live);
    require(policy.current().accepted_sequence == 0 && policy.current().applied_uv == 0, "New scope kept old plane");
    require(!policy.observe(required(1250, 2), 1300), "Old scope revived placement");
  }

  void generic_rotation_and_logical_changes() {
    adaptive::policy policy;
    adaptive::sample previous;
    for (unsigned frame = 0; frame < 120; ++frame) {
      auto value = required(1000 + 16 * frame, 5);
      value.provenance.generic_basis_epoch = 17;
      value.provenance.generic_routing_epoch = 23;
      value.provenance.source_id = 101 + frame % 3;
      value.provenance.revision = 31 + frame % 3; // Per-allocation layout epochs differ.
      policy.prepare(value.provenance);
      if (frame % 7 == 1)
        require(policy.observe(previous, value.provenance.now_ms), "Off-turn admitted generic readback was rejected");
      previous = value;
    }
    require(near(policy.current().applied_uv, .0035f) && policy.current().target_uv == .0035f,
      "Generic ABC rotation repeatedly reset placement or confirmation");
    auto live = previous.provenance;
    live.now_ms += 16; live.tick_ms = live.now_ms; ++live.sequence; ++live.source_id; ++live.revision;
    policy.prepare(live);
    require(near(policy.current().applied_uv, .0035f), "Physical identity changed logical placement");
    ++live.generic_routing_epoch;
    policy.prepare(live);
    require(policy.current().applied_uv == 0 && !policy.observe(previous, live.now_ms),
      "A different capture group admitted old placement evidence");
    auto other = live; ++other.generic_basis_epoch;
    require(!adaptive::same_scope(live, other), "Generic recalibration retained the old logical scope");
    other = live; ++other.epoch;
    require(!adaptive::same_scope(live, other), "Runtime replacement retained generic scope");
    other = live; other.generic_routing_epoch = 0;
    require(!adaptive::fresh(other) && !adaptive::same_scope(live, other), "Incomplete generic scope was eligible");
    auto provider = source(4000); other = provider; ++other.source_id;
    require(!adaptive::same_scope(provider, other), "Provider feature replacement was treated as buffer rotation");
    other = provider; ++other.revision;
    require(!adaptive::same_scope(provider, other), "Provider observation loss retained scope");
  }

  void central_partition_edges_and_malformed_data() {
    for (const auto extent : std::array<std::array<unsigned, 2>, 5>{{{1, 1}, {2, 2}, {3, 5}, {17, 19}, {8192, 8192}}}) {
      adaptive::policy policy; auto value = sample(1000, extent[0], extent[1]);
      for (auto &cell : value.tiles) { cell.covered = cell.pixels; cell.bad[0] = cell.pixels; }
      feed(policy, value);
      const std::uint64_t expected = (7 * extent[0] / 8 - extent[0] / 8) * (7 * extent[1] / 8 - extent[1] / 8);
      require(policy.current().center_pixels == expected && policy.current().covered_pixels == expected &&
          policy.current().conflict_counts[0] == expected, "Central pixel partition changed");

    }
    adaptive::policy policy; auto value = sample(1000);
    for (unsigned y = 0; y < 16; ++y) for (unsigned x = 0; x < 16; ++x)
      if (!adaptive::central_tile(x, y)) { auto &c = value.tiles[y * 16 + x]; c.covered = 100; c.bad.fill(100); }
    feed(policy, value); require(policy.current().covered_pixels == 0 && policy.current().required_uv == 0, "Edge UI drove placement");
    value.provenance = source(1100); value.tiles[137].covered = 100; value.tiles[137].bad[0] = 21; feed(policy, value);
    value.provenance = source(1200); feed(policy, value);
    require(policy.current().covered_pixels == 100 && policy.current().target_uv == .001f, "Edges diluted the central denominator");
    const auto rejected = [](adaptive::sample data) {
      adaptive::policy p; p.prepare(source(1000)); require(!p.observe(data, 1000) && !p.current().accepted_sequence, "Malformed data admitted");
    };
    value = sample(1000); value.width = 0; rejected(value);
    value = sample(1000); value.width = 8193; rejected(value);
    value = sample(1000); ++value.tiles[0].pixels; rejected(value);
    value = sample(1000); --value.tiles[0].pixels; ++value.tiles[1].pixels; rejected(value);
    value = sample(1000); value.tiles[0].covered = value.tiles[0].pixels + 1; rejected(value);
    value = required(1000, 1); value.tiles[137].bad[1] = 101; rejected(value);
    value = sample(1000); value.tiles[137].covered = value.tiles[137].invalid = 1; rejected(value);
    adaptive::policy late; late.prepare(source(1000)); require(!late.observe(sample(1001), 1000), "Future timestamp admitted");
    late.prepare(source(1300)); require(!late.observe(sample(1000), 1300), "Expired readback admitted");
  }
}
int main() {
  try {
    entry_boundary_and_absolute_levels(); sustained_approach_and_actual_slew(); strict_release_and_conservative_dwell();
    area_rule_catches_diffuse_overlap_without_regions(); exact_area_boundaries_and_both_release_tests();
    caps_and_float_emission(); cap_change_holds_actual_uv(); missing_duplicate_gap_and_clock(); retained_mask_and_scope();
    generic_rotation_and_logical_changes();
    central_partition_edges_and_malformed_data();
    std::puts("PASS: eleven area-or-ratio absolute-level adaptive UI policy groups"); return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
