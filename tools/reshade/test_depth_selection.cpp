// SPDX-License-Identifier: GPL-3.0-only
#include "depth_selection_policy.h"
#include "depth_moments.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
  using namespace sunshine_depth;
  constexpr unsigned width = 32, height = 18;

  void require(bool value, const char *description) {
    if (!value)
      throw std::runtime_error(description);
  }

  std::vector<float> gradient(float low = .02f, float high = .6f) {
    std::vector<float> values(width * height);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        values[y * width + x] = low + (high - low) * (float(x) / (width - 1) * .7f + float(y) / (height - 1) * .3f);
    return values;
  }

  depth_quality analyze(const std::vector<float> &values) {
    return analyze_depth(values.data(), values.size(), width, height);
  }

  struct sampled_scene {
    sunshine_depth_statistics::tile_layout grid;
    std::vector<float> values;

    depth_quality quality() const {
      return analyze_depth(values.data(), values.size(), grid.x, grid.y);
    }
  };

  template<class Depth>
  sampled_scene sample_scene(unsigned crop_width, unsigned crop_height, Depth depth) {
    const auto grid = sunshine_depth_statistics::tile_grid(crop_width, crop_height);
    sampled_scene scene {grid, std::vector<float>(std::size_t(grid.x) * grid.y)};
    for (unsigned y = 0; y < grid.y; ++y)
      for (unsigned x = 0; x < grid.x; ++x) {
        const float u = (sunshine_depth_statistics::tile_bounds(x, crop_width, grid.x).center() + .5f) / crop_width;
        const float v = (sunshine_depth_statistics::tile_bounds(y, crop_height, grid.y).center() + .5f) / crop_height;
        scene.values[std::size_t(y) * grid.x + x] = depth(u, v);
      }
    return scene;
  }

  candidate_info candidate(std::uint64_t id, std::uint64_t frame, std::uint64_t draws, unsigned w = 640, unsigned h = 360) {
    candidate_info result;
    result.id = id;
    result.width = w;
    result.height = h;
    result.viewport_width = float(w);
    result.viewport_height = float(h);
    result.output_width = 1280;
    result.output_height = 720;
    result.last_seen_frame = frame;
    result.draws = draws;
    result.vertices = draws * 3;
    return result;
  }

  void test_content() {
    const auto scene = gradient();
    require(analyze(scene).kind == content_kind::useful, "spatially coherent scene depth must be useful");
    require(analyze(gradient(.99999f, .999999f)).kind == content_kind::useful, "normal far depth cannot need large absolute variance");
    require(analyze(gradient(1.e-9f, 1.001e-9f)).kind == content_kind::useful, "tiny varied reversed-Z depth must be useful");
    require(analyze(gradient(1.e-25f, 2.e-25f)).kind == content_kind::useful, "depth quality must be scale independent");
    for (float value : {0.f, 1.f, .1f, .999999f})
      require(analyze(std::vector<float>(width * height, value)).kind == content_kind::flat, "constant plane/clear depth is not evidence of scene geometry");

    auto values = std::vector<float>(width * height, 1.f);
    for (unsigned y = 0; y < height; ++y)
      values[y * width + 10] = .2f;
    require(analyze(values).kind == content_kind::flat, "white depth with a sparse line must not become positive evidence");
    values = scene;
    values[10] = std::numeric_limits<float>::quiet_NaN();
    require(analyze(values).kind == content_kind::unreliable, "nonfinite depth cannot provide positive evidence");
    values[10] = -1.f;
    require(analyze(values).kind == content_kind::unreliable, "out-of-range depth cannot provide positive evidence");
    values[10] = 1.1f;
    require(analyze(values).kind == content_kind::unreliable, "greater-than-one depth cannot provide positive evidence");
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        values[y * width + x] = (x + y) % 2 ? .2f : .8f;
    require(analyze(values).kind == content_kind::unreliable, "checkerboard garbage cannot win on variance alone");
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        values[y * width + x] = y % 2 ? .2f : .8f;
    require(analyze(values).kind == content_kind::unreliable, "horizontal stripe garbage cannot hide behind horizontal zero differences");
    std::uint32_t random = 1234;
    for (auto &value : values) {
      random = random * 1664525u + 1013904223u;
      value = float(random >> 8) / 16777216.f;
    }
    require(analyze(values).kind == content_kind::unreliable, "uncorrelated noise cannot count as scene depth");
    require(analyze_depth(nullptr, 0, width, height).kind == content_kind::unknown, "missing readback must stay unknown");
    require(analyze_depth(scene.data(), scene.size() - 1, width, height).kind == content_kind::unknown, "incomplete readback must stay unknown");
    std::puts("PASS raw content: normal/reversed tiny depth, clear lines, invalid values, spatial noise, missing readback");
  }

  void test_shapes() {
    require(auto_candidate_shape(2560, 1440, 3840, 2160), "DLSS Quality scaled depth must qualify");
    require(auto_candidate_shape(1920, 1080, 3840, 2160), "DLSS Performance scaled depth must qualify");
    require(auto_candidate_shape(1280, 720, 3840, 2160), "DLSS Ultra Performance scaled depth must qualify");
    require(auto_candidate_shape(3840, 2160, 3840, 2160), "native TAA depth must qualify");
    require(auto_candidate_shape(960, 540, 3840, 2160), "quarter-resolution depth must qualify");
    require(!auto_candidate_shape(1024, 1024, 3840, 2160), "square shadow map must not pass screen shape");
    require(!auto_candidate_shape(3840, 128, 3840, 2160), "skinny shadow map must not pass screen shape");
    require(!auto_candidate_shape(640, 360, 3840, 2160), "very small target must not pass default shape");
    require(!auto_candidate_shape(0, 0, 3840, 2160), "zero target dimensions must not qualify");
    std::puts("PASS automatic shape: DLSS scales, native TAA, square/skinny/tiny exclusions");
  }

  void test_workload_and_hysteresis() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    const auto flat = analyze(std::vector<float>(width * height, 1.f));
    policy.observe(candidate(1, 10, 10000, 1280, 720));
    policy.observe(candidate(2, 10, 2));
    require(policy.choose(10).id == 1, "startup without content should preserve conventional workload fallback");
    policy.sample(1, flat, 10);
    policy.sample(2, useful, 10);
    require(policy.choose(10, 1).id == 1, "one promising capture cannot cause a switch");
    policy.sample(2, useful, 11);
    require(policy.choose(11, 1).id == 2, "confirmed low-draw scene must beat high-draw flat depth");
    policy.sample(1, useful, 11);
    policy.sample(1, useful, 12);
    require(policy.choose(12, 2).id == 2, "higher workload cannot dislodge current confirmed scene");

    // Same resource dimensions now carry different contents: three bad captures
    // revoke confidence, then another confirmed source can replace the old one.
    policy.sample(2, flat, 12);
    require(policy.choose(12, 2).id == 2, "one bad scene capture must be tolerated");
    policy.sample(2, flat, 13);
    require(policy.choose(13, 2).id == 2, "two bad scene captures must be tolerated");
    policy.observe(candidate(1, 14, 10000, 640, 360));
    policy.observe(candidate(2, 14, 2, 640, 360));
    policy.sample(2, flat, 14);
    require(policy.choose(14, 2).id == 1, "sustained bad content must yield to confirmed same-size replacement");
    std::puts("PASS ranking: low-draw scene wins, single noise ignored, same-size contents recover with hysteresis");
  }

  void test_flat_menu_and_unknown() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    const auto flat = analyze(std::vector<float>(width * height, 0.f));
    for (std::uint64_t id : {1u, 2u, 3u})
      policy.observe(candidate(id, 10, id * 100));
    policy.sample(1, useful, 10);
    policy.sample(1, useful, 11);
    require(policy.choose(11, 1).id == 1, "test scene should begin confirmed");
    for (std::uint64_t frame = 12; frame < 30; ++frame) {
      for (std::uint64_t id : {1u, 2u, 3u}) {
        policy.observe(candidate(id, frame, id * frame * 100));
        policy.sample(id, flat, frame);
      }
      require(policy.choose(frame, 1).id == 1, "flat menu must not oscillate through high-workload buffers");
    }
    policy.sample(1, depth_quality {}, 30);
    require(policy.evidence(1)->sampled_frame == 29, "missing readback must not overwrite actual content evidence");
    policy.sample(2, useful, 30);
    policy.sample(2, flat, 31);
    policy.sample(2, useful, 32);
    require(!policy.evidence(2)->confirmed_good, "alternating useful/noisy content cannot accumulate confirmation");
    std::puts("PASS menus: stable selection across flat contents and missing/alternating samples");
  }

  void test_lifetime_and_fairness() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    for (std::uint64_t id : {11u, 2u, 7u})
      policy.observe(candidate(id, 10, id == 2 ? 1 : 10000));
    for (std::uint64_t expected : {2u, 7u, 11u, 2u, 7u, 11u})
      require(policy.next_probe(10, 11) == expected, "round-robin must visit low-draw candidate and current fairly");
    policy.sample(2, useful, 10);
    policy.sample(2, useful, 11);
    require(policy.choose(11, 7).id == 2, "candidate should have established positive confidence");
    policy.erase(2);
    policy.observe(candidate(20, 11, 1)); // Same native pointer may now have this new generation.
    policy.sample(2, useful, 12); // Late result for the destroyed generation.
    require(policy.evidence(2) == nullptr, "late readback must not resurrect destroyed candidate");
    require(!policy.evidence(20)->confirmed_good, "resource-address reuse must not inherit prior confidence");
    require(policy.choose(12, 2).id != 2, "destroyed selected candidate cannot remain selected");
    policy.sample(20, useful, 12);
    policy.sample(20, useful, 12);
    require(!policy.evidence(20)->confirmed_good, "duplicate same-frame capture cannot establish confidence");
    require(policy.next_probe(100) == 0 && policy.choose(100, 11).id == 0, "inactive candidate metadata cannot remain selectable/probeable");
    std::puts("PASS lifetime: fair exploration, destruction, pointer reuse, delayed/duplicate samples, stale activity");
  }

  void test_staleness_and_order() {
    policy_options options;
    options.evidence_stale_frames = 5;
    selection_policy policy(options);
    const auto useful = analyze(gradient());
    for (std::uint64_t id : {9u, 3u}) {
      policy.observe(candidate(id, 10, 100));
      policy.sample(id, useful, 9);
      policy.sample(id, useful, 10);
    }
    require(policy.choose(10).id == 3, "ties must use stable lifetime ordering independent of insertion");
    policy.observe(candidate(9, 17, 100));
    policy.observe(candidate(3, 17, 100));
    policy.sample(9, useful, 16);
    require(!policy.evidence(9)->confirmed_good, "one fresh capture cannot renew expired positive evidence");
    policy.sample(9, useful, 17);
    require(policy.choose(17, 3).id == 9, "stale historical confidence cannot beat recent confirmed contents");
    policy.sample(9, analyze(std::vector<float>(width * height, 1.f)), 15);
    require(policy.evidence(9)->sampled_frame == 17, "out-of-order readback cannot replace newer evidence");
    std::puts("PASS evidence freshness and deterministic total ordering");
  }

  void test_late_native_probe_priority() {
    for (const unsigned output_width : {1920u, 3840u}) {
      selection_policy policy;
      const unsigned output_height = output_width * 9 / 16;
      const auto source = [&](std::uint64_t id, bool native) {
        auto info = candidate(id, 10, 1, native ? output_width : output_width / 2,
                              native ? output_height : output_height / 2);
        info.output_width = output_width;
        info.output_height = output_height;
        return info;
      };
      for (std::uint64_t id = 1; id <= 24; ++id)
        policy.observe(source(id, false));
      require(policy.next_probe(10, 1) == 1 && policy.next_probe(10, 1) == 2,
              "without native candidates the ordinary round-robin must be unchanged");
      policy.observe(source(100, true));
      require(policy.next_probe(10, 1) == 100,
              "a late native source must be probed before the older low-resolution backlog");
      require(policy.next_probe(10, 1) == 3,
              "priority must not advance or restart the ordinary round-robin cursor");
      require(!policy.evidence(100)->confirmed_good,
              "a priority visit must not manufacture content qualification");
    }
    std::puts("PASS native discovery: late 1080p/4K output-matching source gets the next visit without losing ordinary progress");
  }

  void test_native_probe_fairness() {
    selection_policy policy;
    for (std::uint64_t id : {1u, 2u, 3u})
      policy.observe(candidate(id, 10, 1000));
    for (std::uint64_t id : {100u, 101u, 102u})
      policy.observe(candidate(id, 10, 1, 1280, 720));
    const std::array<std::uint64_t, 6> ordinary {1, 2, 3, 100, 101, 102};
    for (std::size_t visit = 0; visit < ordinary.size() * 2; ++visit) {
      require(policy.next_probe(10, 1) == 100 + visit % 3,
              "native priority must rotate through every full-output candidate");
      require(policy.next_probe(10, 1) == ordinary[visit % ordinary.size()],
              "each native visit must leave an ordinary visit for every active candidate");
    }
    policy.erase(100);
    require(policy.next_probe(10, 1) == 101,
            "destroyed priority source must not be returned when its lane wraps");
    std::puts("PASS native discovery fairness: multiple native candidates alternate with a complete ordinary scan, destroyed IDs excluded");
  }

  void test_unusable_native_probe_fallback() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    const auto flat = analyze(std::vector<float>(width * height, .5f));
    for (std::uint64_t id : {1u, 2u, 3u})
      policy.observe(candidate(id, 10, 1));
    policy.observe(candidate(100, 10, 10000, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      policy.sample(1, useful, frame);
      policy.sample(100, flat, frame);
    }
    for (unsigned round = 0; round < 3; ++round) {
      for (std::uint64_t ordinary : {1u, 2u, 3u, 100u}) {
        require(policy.next_probe(10, 1) == 100, "a flat native candidate may still receive a bounded discovery visit");
        require(policy.next_probe(10, 1) == ordinary, "unusable native contents must not starve lower-resolution exploration");
        require(policy.choose(10, 1).id == 1, "native probing must not bypass useful-content selection");
      }
    }
    std::puts("PASS native discovery fallback: unusable full-resolution source cannot starve other sources or replace useful selected depth");
  }

  void test_native_probe_activity_and_coverage() {
    selection_policy policy;
    policy.observe(candidate(1, 10, 1));
    policy.observe(candidate(2, 10, 1, 2560, 1440)); // Oversized, not an output-sized viewport.
    auto unknown_viewport = candidate(3, 10, 1, 2560, 1440);
    unknown_viewport.viewport_width = unknown_viewport.viewport_height = 0;
    policy.observe(unknown_viewport);
    policy.observe(candidate(4, 10, 0, 1280, 720)); // Clear-only.
    policy.observe(candidate(5, 6, 1, 1280, 720));  // Outside the activity grace.
    policy.observe(candidate(6, 11, 1, 1280, 720)); // Future activity.
    for (std::uint64_t expected : {1u, 2u, 3u, 1u, 2u, 3u})
      require(policy.next_probe(10, 1) == expected,
              "unknown oversized viewport and inactive native candidates cannot acquire priority");
    auto padded = candidate(20, 10, 1, 2560, 1440);
    padded.viewport_width = 1280;
    padded.viewport_height = 720;
    policy.observe(padded);
    require(policy.next_probe(10, 1) == 20,
            "an authoritative output-sized viewport in a padded allocation must use the same native coverage rule");
    std::puts("PASS native discovery guards: clear-only, stale, future and unproven oversized targets excluded; valid padded viewport supported");
  }

  void test_native_incumbent_probe_order() {
    selection_policy policy;
    policy.observe(candidate(1, 10, 1));
    policy.observe(candidate(2, 10, 1));
    policy.observe(candidate(100, 10, 1, 1280, 720));
    for (std::uint64_t expected : {1u, 2u, 100u, 1u, 2u, 100u})
      require(policy.next_probe(10, 100) == expected,
              "a native incumbent must retain the ordinary exploration schedule");
    require(policy.next_probe(10) == 100,
            "startup without an anchor must test an output-matching source early");
    require(policy.next_probe(10) == 1,
            "startup native priority must retain an ordinary fallback visit");
    policy.observe(candidate(200, 10, 10000, 2560, 1440));
    require(policy.next_probe(10, 200) == 100,
            "an unproven oversized incumbent must not block native discovery");
    std::puts("PASS native incumbent and startup: native selection preserves ordinary scan; absent or oversized anchor permits priority");
  }

  void test_clear_only_activity() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    policy.observe(candidate(1, 100, 1));
    policy.sample(1, useful, 99);
    policy.sample(1, useful, 100);
    require(policy.choose(100).id == 1, "rendered scene must qualify before a clear-only frame");

    // The automatic loop observes every live candidate, including an incumbent
    // whose frame only cleared it. Cached positive captures cannot turn that
    // clear into scene work.
    policy.observe(candidate(1, 101, 0));
    require(policy.choose(101, 1).id == 0, "clear-only current source cannot remain an active automatic candidate");
    require(policy.next_probe(101) == 0, "clear-only source cannot schedule a content capture");
    policy.sample(1, useful, 101); // Late preserved pixels still cannot establish current rendering.
    require(policy.choose(101, 1).id == 0, "new readback cannot substitute for a current scene draw");

    policy.observe(candidate(2, 101, 1));
    policy.sample(2, useful, 100);
    policy.sample(2, useful, 101);
    require(policy.choose(101, 1).id == 2, "current useful replacement must outrank a clear-only incumbent");
    policy.observe(candidate(1, 102, 1));
    require(policy.choose(102, 1).id == 1, "resumed rendering must restore the retained scene without inventing a new lifetime");
    std::puts("PASS clear-only activity: no ready candidate or probe, no late-readback substitute for a draw, fresh replacement and resumed rendering");
  }

  depth_quality quality_score(float score) {
    auto quality = analyze(gradient());
    quality.score = score;
    return quality;
  }

  selection_result paired_challenge(selection_policy &policy, std::uint64_t first,
                                    candidate_info incumbent, const depth_quality &current_quality,
                                    candidate_info challenger, const depth_quality &challenger_quality) {
    selection_result selected;
    for (unsigned round = 0; round < 3; ++round) {
      incumbent.last_seen_frame = challenger.last_seen_frame = first + round;
      policy.observe(incumbent);
      policy.observe(challenger);
      policy.sample(incumbent.id, current_quality, first + round);
      policy.sample(challenger.id, challenger_quality, first + round);
      selected = policy.choose(first + round, incumbent.id);
      require(selected.id == (round == 2 ? challenger.id : incumbent.id),
              "healthy replacement needs three fresh paired wins, not merely three useful challenger captures");
    }
    return selected;
  }

  void test_qualified_native_challenge() {
    selection_policy policy;
    policy.observe(candidate(1, 10, 1));
    policy.observe(candidate(2, 10, 1, 1280, 720));
    // Replay the relative quality observed in Dead Space: the smaller source is
    // slightly smoother, while both native and smaller sources pass validation.
    const auto smaller = quality_score(.976f), native = quality_score(.952f);
    for (std::uint64_t frame : {8u, 9u, 10u}) policy.sample(1, smaller, frame);
    policy.sample(2, native, 10);
    policy.sample(2, native, 11);
    require(policy.evidence(2)->confirmed_good, "native candidate should already satisfy basic content qualification");
    for (unsigned repeat = 0; repeat < 100; ++repeat)
      require(policy.choose(11, 1).id == 1, "per-frame choose calls must not manufacture a third independent capture");
    policy.sample(2, native, 12);
    paired_challenge(policy, 12, candidate(1, 12, 1), smaller, candidate(2, 12, 1, 1280, 720), native);
    require(policy.source_coverage(1) == .5f && policy.source_coverage(2) == 1.f, "coverage must describe effective source scale");
    require(policy.comparison_score(2) >= policy.retention_score(1) + .04, "native challenge must clear a meaningful bounded score margin");
    require(policy.choose(14, 2).id == 2, "native selection must not immediately switch back to smoother lower resolution");
    std::puts("PASS qualified challenge: replay-like native vs smoother half source, three distinct captures, meaningful margin");
  }

  void test_equal_size_and_quality_blips() {
    const auto strong = quality_score(.976f), weak = quality_score(.6f), flat = analyze(std::vector<float>(width * height, 1.f));
    selection_policy equal;
    equal.observe(candidate(1, 10, 1, 1280, 720));
    equal.observe(candidate(2, 10, 1000000, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      equal.sample(1, quality_score(.8f), frame);
      equal.sample(2, quality_score(.99f), frame);
    }
    for (unsigned repeat = 0; repeat < 100; ++repeat)
      require(equal.choose(10, 1).id == 1, "equal-size content and workload differences cannot cause marginal switching");

    selection_policy policy;
    policy.observe(candidate(1, 10, 1));
    policy.observe(candidate(2, 10, 10000, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      policy.sample(1, strong, frame);
      policy.sample(2, weak, frame);
    }
    require(policy.choose(10, 1).id == 1, "substantially less credible native contents cannot win on size");
    policy.sample(1, weak, 11);
    require(policy.choose(11, 1).id == 1, "one weak incumbent sample must not cheapen its replacement threshold");
    policy.sample(1, flat, 12);
    require(policy.choose(12, 1).id == 1, "one bad incumbent sample must not enable an unrelated native switch");
    policy.sample(2, strong, 11);
    require(policy.choose(12, 1).id == 1, "one high-quality challenger blip cannot hide its weaker recent captures");
    policy.sample(2, strong, 12);
    require(policy.choose(12, 1).id == 1, "two improved captures still retain one weaker recent challenger sample");
    policy.sample(2, strong, 13);
    paired_challenge(policy, 13, candidate(1, 13, 1), strong, candidate(2, 13, 10000, 1280, 720), strong);
    std::puts("PASS comparison hysteresis: equal-size stability, incumbent weakness protection, sustained challenger quality");
  }

  void test_native_is_not_validity() {
    const auto useful = quality_score(.98f), flat = analyze(std::vector<float>(width * height, 1.f));
    auto noisy = useful;
    noisy.kind = content_kind::unreliable;
    selection_policy policy;
    policy.observe(candidate(1, 10, 1));
    policy.observe(candidate(2, 10, 1000000, 1280, 720));
    policy.sample(1, useful, 9);
    policy.sample(1, useful, 10);
    for (std::uint64_t frame : {8u, 9u, 10u}) policy.sample(2, flat, frame);
    require(policy.choose(10, 2).id == 1, "useful lower resolution must replace bad native even with only two positive captures");
    for (std::uint64_t frame : {11u, 12u, 13u}) {
      policy.observe(candidate(1, frame, 1));
      policy.observe(candidate(2, frame, 1000000, 1280, 720));
      policy.sample(2, noisy, frame);
      require(policy.choose(frame, 1).id == 1, "noisy native must never outrank useful lower resolution");
    }
    policy.sample(2, depth_quality {}, 14);
    require(policy.choose(14, 1).id == 1, "missing native readback must not imply positive evidence");

    selection_policy oversized;
    oversized.observe(candidate(1, 10, 1));
    oversized.observe(candidate(3, 10, 1000000, 2560, 1440));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      oversized.sample(1, useful, frame);
      oversized.sample(3, useful, frame);
    }
    require(oversized.source_coverage(3) == 0 && oversized.choose(10, 1).id == 1, "oversized allocation must not earn a native-coverage reward");
    auto padded = candidate(3, 11, 1, 2560, 1440);
    padded.viewport_width = padded.viewport_height = 0;
    oversized.observe(padded);
    require(oversized.source_coverage(3) == 0, "unknown active viewport cannot turn an oversized allocation into native depth");
    padded.viewport_width = 1280;
    padded.viewport_height = 720;
    oversized.observe(padded);
    require(oversized.source_coverage(3) == 1, "an explicitly valid native viewport in a padded allocation may earn native coverage");
    std::puts("PASS validity prerequisite: bad/noisy/missing native loses, oversized allocation gets no reward without valid native viewport");
  }

  void test_comparison_capture_freshness() {
    policy_options options;
    options.evidence_stale_frames = 5;
    selection_policy policy(options);
    const auto useful = quality_score(.98f);
    policy.observe(candidate(1, 23, 1));
    policy.observe(candidate(2, 23, 1, 1280, 720));
    for (std::uint64_t frame : {20u, 21u, 22u}) policy.sample(1, useful, frame);
    for (std::uint64_t frame : {17u, 18u, 22u}) policy.sample(2, useful, frame);
    require(policy.evidence(2)->good_samples == 3 && policy.evidence(2)->confirmed_good, "test challenger must have three historical positive captures");
    require(policy.choose(23, 1).id == 1, "all three comparison captures must be fresh, not just the last one");
    policy.sample(2, useful, 23);
    paired_challenge(policy, 23, candidate(1, 23, 1), useful, candidate(2, 23, 1, 1280, 720), useful);
    policy.erase(2);
    policy.observe(candidate(12, 25, 1, 1280, 720));
    policy.sample(2, useful, 26);
    require(policy.choose(26, 1).id == 1, "late native score cannot transfer to a new resource lifetime");
    std::puts("PASS comparison freshness: three fresh captures, no delayed evidence transfer across resource lifetimes");
  }

  std::vector<float> people_depth(float endpoint = 0, float left = .35f, float right = .65f, float top = .1f, float bottom = .9f) {
    auto values = gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) {
        const float u = (x + .5f) / width, v = (y + .5f) / height;
        if (u < left || u > right || v < top || v > bottom)
          values[y * width + x] = endpoint;
      }
    return values;
  }

  std::vector<float> sky_depth(float sky_fraction, float endpoint = 0, bool tiny = false) {
    auto values = tiny ? gradient(1.e-12f, 2.e-12f) : gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if ((y + .5f) / height < sky_fraction)
          values[y * width + x] = endpoint;
    return values;
  }

  void test_completeness_measurement_and_noise() {
    const auto people = analyze(people_depth()), white_people = analyze(people_depth(1));
    const auto scene = analyze(gradient(.01f, .045f));
    require(people.kind == content_kind::useful && white_people.kind == content_kind::useful, "coherent people-only depth is valid partial input, not flat garbage");
    require(people.nonendpoint_fraction > .20f && people.nonendpoint_fraction < .30f, "people fixture should occupy roughly one quarter of the depth image");
    require(people.spatial_coverage < scene.spatial_coverage && people.completeness + .20f < scene.completeness, "full scene must carry materially more spatial support than the same-size person island");
    require(people.completeness == white_people.completeness, "both exact endpoint conventions must produce the same occupancy estimate");
    require(scene.nonendpoint_fraction == 1 && scene.spatial_coverage == 1 && scene.completeness == 1, "full coherent scene must occupy the complete sampled support");

    auto values = gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if (x % 2) values[y * width + x] = 0;
    require(analyze(values).kind == content_kind::unreliable, "alternating depth/endpoint columns cannot earn completeness from vertical-only support");
    values = gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if (y % 2) values[y * width + x] = 1;
    require(analyze(values).kind == content_kind::unreliable, "alternating depth/endpoint rows cannot earn completeness from horizontal-only support");
    values = gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if (x != width / 2) values[y * width + x] = 1;
    require(analyze(values).kind != content_kind::useful, "one varying depth column is not coherent two-dimensional scene support");

    values.assign(width * height, 0);
    std::uint32_t random = 9123;
    for (auto &value : values) {
      random = random * 1664525u + 1013904223u;
      if (random % 5 == 0) value = .2f + .6f * float(random >> 8) / 16777216.f;
    }
    const auto speckles = analyze(values);
    require(speckles.nonendpoint_fraction > .15f && speckles.nonendpoint_fraction < .25f, "noise fixture should contain distributed twenty-percent nonendpoint samples");
    require(speckles.kind == content_kind::unreliable, "scattered endpoint-background speckles must not win through coarse tile occupancy");
    std::puts("PASS completeness metrics: people/full scene, both endpoints, white lines, alternating support, scattered speckles");
  }

  void test_full_scene_beats_people() {
    const auto people = analyze(people_depth()), scene = analyze(gradient(.01f, .045f));
    selection_policy same_size;
    same_size.observe(candidate(1, 10, 1, 1280, 720));
    same_size.observe(candidate(2, 10, 1, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) same_size.sample(1, people, frame);
    same_size.sample(2, scene, 10);
    same_size.sample(2, scene, 11);
    for (unsigned i = 0; i < 100; ++i)
      require(same_size.choose(11, 1).id == 1, "fuller same-size source needs three real captures, not repeated choose calls");
    same_size.sample(2, scene, 12);
    paired_challenge(same_size, 12, candidate(1, 12, 1, 1280, 720), people, candidate(2, 12, 1, 1280, 720), scene);

    selection_policy lower_resolution;
    lower_resolution.observe(candidate(1, 10, 1000, 1280, 720));
    lower_resolution.observe(candidate(2, 10, 1, 640, 360));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      lower_resolution.sample(1, people, frame);
      lower_resolution.sample(2, scene, frame);
    }
    paired_challenge(lower_resolution, 10, candidate(1, 10, 1000, 1280, 720), people, candidate(2, 10, 1), scene);
    require(lower_resolution.choose(12, 2).id == 2, "partial native must not steal selection back from a complete smaller source");

    selection_policy worse_quality;
    worse_quality.observe(candidate(1, 10, 1, 1280, 720));
    worse_quality.observe(candidate(2, 10, 1, 1280, 720));
    auto reliable_partial = people, borderline_complete = scene;
    // Policy-level confidence inputs isolate the common quality guard from the
    // analyzer: fuller coverage must not waive substantially weaker coherence.
    reliable_partial.score = .90f;
    borderline_complete.score = .35f;
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      worse_quality.sample(1, reliable_partial, frame);
      worse_quality.sample(2, borderline_complete, frame);
    }
    require(worse_quality.choose(10, 1).id == 1, "meaningfully poorer full-support quality cannot displace reliable partial contents");
    std::puts("PASS fuller-source preference: same resolution, lower-resolution complete scene, global quality guard");
  }

  void test_aspect_grid_candidate_quality() {
    const auto surface = [](float u, float v) { return .01f + .035f * (.7f * u + .3f * v); };
    // Exercise the sampler's actual integer pixel centers, including dimensions
    // that do not divide evenly into tiles, instead of resizing a fixed 32x18 grid.
    for (const auto dimensions : {std::array<unsigned, 2> {1920, 1080}, {1600, 1200}, {3440, 1440}, {1080, 1920}}) {
      const auto crop_width = dimensions[0], crop_height = dimensions[1];
      const auto full = sample_scene(crop_width, crop_height, surface);
      const auto larger = sample_scene(2 * crop_width, 2 * crop_height, surface);
      const auto scene = full.quality(), high_resolution = larger.quality();
      require(full.grid.x == larger.grid.x && full.grid.y == larger.grid.y,
              "equal-aspect higher resolution must retain the same candidate sampling layout");
      require(scene.kind == content_kind::useful && high_resolution.kind == content_kind::useful,
              "coherent scene depth must remain useful at landscape, ultrawide and portrait aspects");
      require(scene.completeness == 1 && high_resolution.completeness == 1,
              "full scene support must not depend on tile dimensions or source resolution");
      require(std::abs(scene.score - high_resolution.score) < .002f &&
              std::abs(scene.histogram_span - high_resolution.histogram_span) < .01f,
              "equivalent higher-resolution geometry must preserve candidate quality and histogram breadth");

      for (float endpoint : {0.f, 1.f}) {
        const auto people = sample_scene(crop_width, crop_height, [&](float u, float v) {
          return u >= .35f && u <= .65f && v >= .1f && v <= .9f ? surface(u, v) : endpoint;
        }).quality();
        require(people.kind == content_kind::useful && people.completeness + .20f < scene.completeness,
                "aspect-aware sampling must retain coherent partial depth while recognizing fuller scene support");
        const auto terrain = sample_scene(crop_width, crop_height, [&](float u, float v) {
          return u < .75f && v > .9f ? surface(u, v) : endpoint;
        }).quality();
        require(terrain.kind == content_kind::useful && terrain.clear_fraction > .88f,
                "sparse off-center terrain must remain useful despite endpoint-heavy sky at every aspect");
        selection_policy policy;
        auto partial_source = candidate(1, 10, 1, crop_width, crop_height);
        auto full_source = candidate(2, 10, 1, crop_width, crop_height);
        partial_source.output_width = full_source.output_width = crop_width;
        partial_source.output_height = full_source.output_height = crop_height;
        policy.observe(partial_source);
        policy.observe(full_source);
        for (std::uint64_t frame : {8u, 9u, 10u}) {
          policy.sample(1, people, frame);
          policy.sample(2, scene, frame);
        }
        paired_challenge(policy, 10, partial_source, people, full_source, scene);
      }

      auto noise = full;
      for (unsigned y = 0; y < noise.grid.y; ++y)
        for (unsigned x = 0; x < noise.grid.x; ++x)
          noise.values[std::size_t(y) * noise.grid.x + x] = (x + y) % 2 ? .2f : .8f;
      require(noise.quality().kind == content_kind::unreliable,
              "aspect-aware candidate sampling must still reject spatially incoherent depth");
      for (float constant : {0.f, .1f, 1.f}) {
        const auto flat = sample_scene(crop_width, crop_height, [constant](float, float) { return constant; });
        require(flat.quality().kind == content_kind::flat,
                "clear buffers and constant planes must remain flat at every sampling aspect");
      }
      require(analyze_depth(full.values.data(), full.values.size() - 1, full.grid.x, full.grid.y).kind == content_kind::unknown,
              "a truncated dynamic-grid readback must not become candidate evidence");
      require(analyze_depth(full.values.data(), full.values.size(), 0, full.grid.y).kind == content_kind::unknown &&
              analyze_depth(full.values.data(), full.values.size(), full.grid.x, 0).kind == content_kind::unknown,
              "invalid dynamic-grid dimensions must stay unknown");
    }
    require(sample_scene(0, 1080, surface).quality().kind == content_kind::unknown &&
            sample_scene(1920, 0, surface).quality().kind == content_kind::unknown,
            "empty source dimensions cannot produce positive depth evidence");
    std::puts("PASS aspect-aware candidates: resolution-stable quality, full/partial ordering, noise/flat rejection, incomplete readbacks");
  }

  void test_completeness_hysteresis() {
    const auto people = analyze(people_depth()), scene = analyze(gradient(.01f, .045f));
    const auto slightly_larger = analyze(people_depth(0, .25f, .75f, .05f, .95f));
    require(slightly_larger.completeness > people.completeness && slightly_larger.completeness < people.completeness + .20f, "comparison fixture must improve support by less than the material completeness margin");
    selection_policy marginal;
    marginal.observe(candidate(1, 10, 1, 1280, 720));
    marginal.observe(candidate(2, 10, 1000000, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      marginal.sample(1, people, frame);
      marginal.sample(2, slightly_larger, frame);
    }
    require(marginal.comparison_score(2) > marginal.retention_score(1) + .04, "weighted score alone would switch this deliberately subthreshold completeness change");
    require(marginal.choose(10, 1).id == 1, "weighted fallback must not bypass the explicit same-size completeness margin");

    selection_policy transient;
    transient.observe(candidate(1, 10, 1, 1280, 720));
    transient.observe(candidate(2, 10, 1, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      transient.sample(1, people, frame);
      transient.sample(2, people, frame);
    }
    for (std::uint64_t frame = 11; frame <= 17; ++frame) {
      transient.observe(candidate(1, frame, 1, 1280, 720));
      transient.observe(candidate(2, frame, 1, 1280, 720));
      transient.sample(1, people, frame);
      transient.sample(2, frame == 12 ? people : scene, frame);
      require(transient.choose(frame, 1).id == (frame == 17 ? 2u : 1u), "fuller history must qualify, then win three fresh paired comparisons");
    }

    selection_policy incumbent_history;
    incumbent_history.observe(candidate(1, 10, 1, 1280, 720));
    incumbent_history.observe(candidate(2, 10, 1, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      incumbent_history.sample(1, scene, frame);
      incumbent_history.sample(2, scene, frame);
    }
    for (std::uint64_t frame = 11; frame <= 15; ++frame) {
      incumbent_history.observe(candidate(1, frame, 1, 1280, 720));
      incumbent_history.observe(candidate(2, frame, 1, 1280, 720));
      incumbent_history.sample(1, people, frame);
      incumbent_history.sample(2, scene, frame);
      require(incumbent_history.choose(frame, 1).id == (frame == 15 ? 2u : 1u), "old fuller incumbent history must expire before three fresh relative wins can replace it");
    }
    std::puts("PASS completeness hysteresis: explicit minimum gain, transient fuller candidate, incumbent support history ages out");
  }

  void test_sky_support() {
    for (float endpoint : {0.f, 1.f}) {
      for (float sky : {.35f, .70f}) {
        const auto terrain = analyze(sky_depth(sky, endpoint));
        const auto tiny_terrain = analyze(sky_depth(sky, endpoint, true));
        require(terrain.kind == content_kind::useful && tiny_terrain.kind == content_kind::useful, "sky-heavy coherent terrain must remain useful for either endpoint and tiny reversed-Z values");
        require(terrain.completeness == tiny_terrain.completeness, "very small reversed-Z values must count as occupied support without an absolute epsilon");
      }
    }
    auto sliver = gradient(.01f, .045f);
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        if (y < height - 2 || x >= 24) sliver[y * width + x] = 1;
    const auto sky_sliver = analyze(sliver);
    require(sky_sliver.clear_fraction > .90f && sky_sliver.kind == content_kind::useful, "ninety-percent endpoint occupancy alone must not reject coherent two-dimensional terrain");

    selection_policy stable_sky;
    stable_sky.observe(candidate(1, 10, 1, 1280, 720));
    stable_sky.observe(candidate(2, 10, 1000, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      stable_sky.sample(1, analyze(sky_depth(.70f, 1)), frame);
      stable_sky.sample(2, analyze(people_depth()), frame);
    }
    require(stable_sky.choose(10, 1).id == 1, "a coherent sky-heavy scene must not be discarded for a similarly sparse people-only source");
    stable_sky.sample(1, analyze(std::vector<float>(width * height, 1.f)), 11);
    require(stable_sky.choose(11, 1).id == 1, "one all-sky frame must not discard the incumbent");
    std::puts("PASS legitimate sky: endpoint0/1, tiny reversed-Z terrain, coherent low occupancy, stable partial-scene comparison");
  }

  std::vector<float> exponential_depth(float first_exponent = -10, float exponent_span = 8) {
    auto values = gradient(0, 1);
    for (auto &value : values) value = std::exp2(first_exponent + exponent_span * value);
    return values;
  }

  std::vector<float> log_odds_depth(double first_octave, double octave_span) {
    auto values = gradient(0, 1);
    for (auto &value : values) {
      const double odds = std::exp2(first_octave + octave_span * value);
      value = float(odds / (1 + odds));
    }
    return values;
  }

  void test_histogram_measurement() {
    const auto narrow_values = gradient(.04f, .052f);
    const auto narrow = analyze(narrow_values), broad = analyze(exponential_depth());
    require(narrow.kind == content_kind::useful && broad.kind == content_kind::useful, "histogram breadth must not turn coherent narrow or broad scenes into a validity gate");
    require(broad.histogram_span > narrow.histogram_span + .75f && broad.histogram_populated_bins >= narrow.histogram_populated_bins + 2, "exponential scene must have materially wider robust distribution and more populated global bins");
    require(broad.histogram_breadth > narrow.histogram_breadth && broad.histogram_breadth <= 1 && narrow.histogram_breadth >= 0, "histogram diagnostic rank must be bounded and distinguish actual breadth");

    auto complemented = exponential_depth();
    for (auto &value : complemented) value = 1 - value;
    const auto normal = analyze(complemented);
    require(normal.kind == content_kind::useful && std::abs(normal.histogram_span - broad.histogram_span) < .04f, "ordinary representable normal/reversed counterparts must preserve histogram breadth");
    require(normal.histogram_populated_bins == broad.histogram_populated_bins, "global reflected bins must preserve populated-bin count for the complement fixture");
    const auto tiny = analyze(exponential_depth(-100, 8));
    require(tiny.kind == content_kind::useful && tiny.nonendpoint_fraction == 1 && tiny.histogram_span > 3, "tiny positive reversed values must retain globally measured breadth without endpoint epsilon");
    const auto boundary_a = analyze(log_odds_depth(-4.02, .49));
    const auto boundary_b = analyze(log_odds_depth(-3.77, .49));
    require(boundary_a.kind == content_kind::useful && boundary_b.kind == content_kind::useful, "narrow global-bin boundary shifts must remain valid depth");
    require(std::abs(boundary_a.histogram_span - boundary_b.histogram_span) < .001f, "histogram span must not expand merely by translating a narrow distribution across a bin boundary");

    auto outliers = narrow_values;
    outliers[0] = std::numeric_limits<float>::denorm_min();
    outliers.at(width * height - 1) = std::nextafter(1.f, 0.f);
    const auto isolated = analyze(outliers);
    require(isolated.kind == content_kind::useful && std::isfinite(isolated.histogram_span), "representable extreme isolated samples must keep histogram arithmetic and indexing finite");
    require(std::abs(isolated.histogram_span - narrow.histogram_span) < .01f && isolated.histogram_populated_bins == narrow.histogram_populated_bins, "isolated extreme outliers must not expand robust span or populated-bin count");
    for (float endpoint : {0.f, 1.f}) {
      auto sky = exponential_depth();
      for (unsigned y = 0; y < height * 2 / 3; ++y)
        for (unsigned x = 0; x < width; ++x) sky[y * width + x] = endpoint;
      const auto quality = analyze(sky);
      require(quality.kind == content_kind::useful && quality.histogram_span > 2 && quality.histogram_populated_bins >= 3, "legitimate endpoint-heavy sky must retain the terrain histogram");
    }
    std::puts("PASS histogram measurement: global bins, reversal, tiny values, bin phase, endpoint sky, isolated outliers");
  }

  void test_histogram_challenge_and_clawback() {
    const auto narrow = analyze(gradient(.04f, .052f)), broad = analyze(exponential_depth());
    for (bool scaled : {false, true}) {
      selection_policy policy;
      policy.observe(candidate(1, 10, 10000, 1280, 720));
      policy.observe(candidate(2, 10, 1, scaled ? 640 : 1280, scaled ? 360 : 720));
      for (std::uint64_t frame : {8u, 9u, 10u}) policy.sample(1, narrow, frame);
      policy.sample(2, broad, 10);
      policy.sample(2, broad, 11);
      for (unsigned repeat = 0; repeat < 100; ++repeat)
        require(policy.choose(11, 1).id == 1, "broader histogram needs three captures; repeated choose calls cannot manufacture evidence");
      policy.sample(2, broad, 11);
      require(policy.choose(11, 1).id == 1, "duplicate same-frame histogram must not complete challenge history");
      policy.sample(2, broad, 12);
      selection_result selected {1, "native coverage retained"};
      if (!scaled) {
        selected = paired_challenge(policy, 12, candidate(1, 12, 10000, 1280, 720), narrow,
                                    candidate(2, 12, 1, 1280, 720), broad);
        require(selected.id == 2 && std::string(selected.reason).find("histogram") != std::string::npos,
                "sustained broader same-coverage histogram may replace the narrow incumbent");
      }
      for (std::uint64_t frame = 15; frame < 40; ++frame) {
        policy.observe(candidate(1, frame, 10000, 1280, 720));
        policy.observe(candidate(2, frame, 1, scaled ? 640 : 1280, scaled ? 360 : 720));
        policy.sample(1, narrow, frame);
        policy.sample(2, broad, frame);
        require(policy.choose(frame, selected.id).id == selected.id, "histogram breadth alone cannot demote useful native depth or reverse a same-size histogram promotion");
      }
    }
    std::puts("PASS histogram promotion: same-size fresh paired wins, useful native protected from broader lower-resolution content");
  }

  void test_histogram_content_guards() {
    const auto narrow = analyze(gradient(.04f, .052f)), broad = analyze(exponential_depth());
    auto sparse_values = exponential_depth(-14, 12);
    const auto mask = people_depth();
    for (std::size_t i = 0; i < sparse_values.size(); ++i)
      if (mask[i] == 0) sparse_values[i] = 0;
    const auto sparse = analyze(sparse_values);
    require(sparse.kind == content_kind::useful && sparse.histogram_span > narrow.histogram_span + .75f, "partial-person fixture must have a useful and broad depth histogram");
    auto poor = broad;
    poor.score = narrow.score - .30f; // Isolate the policy's common coherence guard.
    auto noisy_values = exponential_depth();
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x) noisy_values[y * width + x] = (x + y) % 2 ? .01f : .9f;
    const auto noise = analyze(noisy_values);
    require(noise.kind == content_kind::unreliable && noise.histogram_span > broad.histogram_span, "noise fixture must be broad yet fail existing coherence validation");
    for (const auto &challenger : {sparse, poor, noise}) {
      selection_policy policy;
      policy.observe(candidate(1, 10, 1));
      policy.observe(candidate(2, 10, 10000, 1280, 720));
      for (std::uint64_t frame : {8u, 9u, 10u}) {
        policy.sample(1, narrow, frame);
        policy.sample(2, challenger, frame);
      }
      require(policy.choose(10, 1).id == 1, "broader sparse people, weaker quality, or noise must not override useful complete depth");
    }

    // Independently isolate both material histogram requirements. A diagnostic
    // score or many populated bins cannot bypass the robust span threshold.
    for (bool sufficient_span : {false, true}) {
      auto insufficient = broad;
      if (sufficient_span) insufficient.histogram_populated_bins = narrow.histogram_populated_bins + 1;
      else insufficient.histogram_span = narrow.histogram_span + .5f;
      selection_policy policy;
      policy.observe(candidate(1, 10, 1, 1280, 720));
      policy.observe(candidate(2, 10, 10000, 1280, 720));
      for (std::uint64_t frame : {8u, 9u, 10u}) {
        policy.sample(1, narrow, frame);
        policy.sample(2, insufficient, frame);
      }
      require(policy.choose(10, 1).id == 1, "histogram promotion must require both material span and genuinely populated-bin gains");
    }
    std::puts("PASS histogram guards: complete support, coherence, quality, robust span and populated-bin requirements");
  }

  void test_histogram_history_and_lifetimes() {
    const auto narrow = analyze(gradient(.04f, .052f)), broad = analyze(exponential_depth());
    selection_policy transient;
    transient.observe(candidate(1, 10, 1, 1280, 720));
    transient.observe(candidate(2, 10, 1, 1280, 720));
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      transient.sample(1, narrow, frame);
      transient.sample(2, narrow, frame);
    }
    for (std::uint64_t frame = 11; frame <= 17; ++frame) {
      transient.observe(candidate(1, frame, 1, 1280, 720));
      transient.observe(candidate(2, frame, 1, 1280, 720));
      transient.sample(1, narrow, frame);
      transient.sample(2, frame == 12 ? narrow : broad, frame);
      require(transient.choose(frame, 1).id == (frame == 17 ? 2u : 1u), "broad history must qualify before three fresh paired wins can promote it");
    }
    for (std::uint64_t frame = 18; frame <= 22; ++frame) {
      transient.observe(candidate(1, frame, 1, 1280, 720));
      transient.observe(candidate(2, frame, 1, 1280, 720));
      transient.sample(1, broad, frame);
      transient.sample(2, narrow, frame);
      require(transient.choose(frame, 2).id == (frame == 22 ? 1u : 2u), "incumbent broad history must age out before three fresh opposite relative wins can replace it");
    }

    policy_options options;
    options.evidence_stale_frames = 5;
    selection_policy freshness(options);
    freshness.observe(candidate(1, 23, 1, 1280, 720));
    freshness.observe(candidate(2, 23, 1, 1280, 720));
    for (std::uint64_t frame : {20u, 21u, 22u}) freshness.sample(1, narrow, frame);
    for (std::uint64_t frame : {17u, 18u, 22u}) freshness.sample(2, broad, frame);
    require(freshness.choose(23, 1).id == 1, "stale oldest histogram capture must not count toward a fresh sustained challenge");
    freshness.sample(2, broad, 23);
    paired_challenge(freshness, 23, candidate(1, 23, 1, 1280, 720), narrow, candidate(2, 23, 1, 1280, 720), broad);
    freshness.erase(2);
    freshness.observe(candidate(12, 25, 1, 1280, 720));
    freshness.sample(2, broad, 26);
    require(freshness.choose(26, 1).id == 1 && freshness.evidence(12)->candidate_histogram_span == 0, "resource lifetime replacement must not inherit old histogram evidence");
    std::puts("PASS histogram history: transient breadth, incumbent ageing, freshness, destroyed and reused resource identities");
  }

  void test_fresh_relative_wins() {
    const auto narrow = analyze(gradient(.04f, .052f)), broad = analyze(exponential_depth());
    selection_policy policy;
    require(policy.pending_challenger_id() == 0, "new policy must not reserve a challenger probe");
    const auto capture = [&](std::uint64_t id, const depth_quality &quality, std::uint64_t frame) {
      policy.observe(candidate(id, frame, 1, 1280, 720));
      policy.sample(id, quality, frame);
    };
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      capture(1, narrow, frame);
      capture(2, broad, frame);
    }
    require(policy.choose(10, 1).id == 1, "three useful candidate captures establish only one relative win");
    require(policy.pending_challenger_id() == 2, "qualified challenger must be exposed for bounded follow-up sampling");
    for (unsigned repeat = 0; repeat < 100; ++repeat) {
      require(policy.choose(10, 1).id == 1, "cached choose calls cannot count as sustained relative superiority");
      require(policy.pending_challenger_id() == 2, "reading the pending challenger must not consume its comparison");
    }
    for (std::uint64_t frame : {11u, 12u, 13u}) {
      capture(2, broad, frame);
      require(policy.choose(frame, 1).id == 1, "challenger-only readbacks cannot replace fresh incumbent comparisons");
    }
    capture(1, narrow, 14);
    require(policy.choose(14, 1).id == 1, "both sources advanced only once since the first paired win");
    require(policy.won_challenger_id() == 0, "a comparison with fewer than three paired wins counted as won");
    capture(2, broad, 14);
    capture(2, broad, 13); // An out-of-order completion must not count.
    require(policy.choose(14, 1).id == 1, "a second challenger sample cannot reuse the incumbent capture");
    capture(1, narrow, 15);
    // The third paired win completes on a Present that only the incumbent
    // rendered: the challenger has no current copy there, so it takes over on
    // the next Present that advances the depth frame with its own work.
    const auto waiting = policy.choose(15, 1);
    require(waiting.id == 1 && std::string(waiting.reason) == "waiting for the challenger's next rendered Present" &&
            policy.pending_challenger_id() == 2, "a won comparison took over on a Present its challenger did not render");
    // The probe keeps its visit to a won challenger, past its deadline, until
    // the takeover adopts its backup; the wait is bounded by the activity
    // grace: a challenger that stops rendering drops the won comparison.
    require(policy.won_challenger_id() == 2, "a won comparison waiting for its challenger's Present was not reported as won");
    {
      auto idle = policy; // The challenger last rendered on frame 13.
      idle.observe(candidate(1, 16, 1, 1280, 720));
      require(idle.choose(16, 1).id == 1 && idle.won_challenger_id() == 2,
              "a won challenger still within the activity grace lost its comparison");
      idle.observe(candidate(1, 17, 1, 1280, 720));
      require(idle.choose(17, 1).id == 1 && idle.won_challenger_id() == 0 && idle.pending_challenger_id() == 0,
              "a won challenger that stopped rendering kept its comparison past the activity grace");
    }
    policy.observe(candidate(2, 15, 1, 1280, 720));
    require(policy.choose(15, 1, false).id == 1, "a won comparison took over on a Present that did not advance the depth frame");
    require(policy.choose(15, 1).id == 2 && policy.won_challenger_id() == 2,
            "three distinct paired wins must eventually promote a better same-size source, still won on its takeover Present");
    require(policy.choose(15, 2).id == 2 && policy.pending_challenger_id() == 0 && policy.won_challenger_id() == 0,
            "adopting the promoted source must release its old pending comparison");
    std::puts("PASS relative evidence: useful captures are not relative wins; cached, duplicate and one-sided samples cannot advance promotion");
  }

  void test_evolving_advantage_and_resets() {
    const auto narrow = analyze(gradient(.04f, .052f)), broad = analyze(exponential_depth());
    const auto flat = analyze(std::vector<float>(width * height, 1.f));
    for (bool bad_without_choose : {false, true}) {
      selection_policy policy;
      const auto capture = [&](std::uint64_t frame, const depth_quality &challenger) {
        policy.observe(candidate(1, frame, 1, 1280, 720));
        policy.observe(candidate(2, frame, 1, 1280, 720));
        policy.sample(1, narrow, frame);
        policy.sample(2, challenger, frame);
      };
      for (std::uint64_t frame : {8u, 9u, 10u}) capture(frame, broad);
      require(policy.choose(10, 1).id == 1, "first relative win must retain source");
      capture(11, broad);
      require(policy.choose(11, 1).id == 1, "second relative win must retain source");
      capture(12, bad_without_choose ? flat : narrow);
      if (!bad_without_choose)
        require(policy.choose(12, 1).id == 1, "lost useful histogram advantage must reset pending wins");
      require(policy.pending_challenger_id() == 0, "lost advantage or bad content must release pending follow-up sampling");
      for (std::uint64_t frame = 13; frame <= 17; ++frame) {
        capture(frame, broad);
        require(policy.choose(frame, 1).id == (frame == 17 ? 2u : 1u),
                "regained advantage must qualify its history then earn three new relative wins");
      }
    }
    for (bool destroy_current : {false, true}) {
      selection_policy policy;
      for (std::uint64_t frame : {8u, 9u, 10u, 11u}) {
        policy.observe(candidate(1, frame, 1, 1280, 720));
        policy.observe(candidate(2, frame, 1, 1280, 720));
        policy.sample(1, narrow, frame);
        policy.sample(2, broad, frame);
        if (frame >= 10) require(policy.choose(frame, 1).id == 1, "pending challenge must not switch early");
      }
      policy.erase(destroy_current ? 1 : 2);
      require(policy.pending_challenger_id() == 0, "destroying either comparison source must release pending follow-up sampling");
      if (destroy_current) {
        require(policy.choose(12, 1).id == 2, "destroyed incumbent must recover immediately to confirmed useful depth");
      } else {
        for (std::uint64_t frame = 12; frame <= 16; ++frame) {
          policy.observe(candidate(1, frame, 1, 1280, 720));
          policy.observe(candidate(3, frame, 1, 1280, 720));
          policy.sample(1, narrow, frame);
          policy.sample(2, broad, frame); // Late completion for destroyed source.
          policy.sample(3, broad, frame);
          require(policy.choose(frame, 1).id == (frame == 16 ? 3u : 1u), "new source lifetime cannot inherit pending wins");
        }
      }
    }
    policy_options options;
    options.evidence_stale_frames = 5;
    selection_policy expired(options);
    for (std::uint64_t frame : {8u, 9u, 10u, 11u, 20u, 21u, 22u, 23u, 24u}) {
      expired.observe(candidate(1, frame, 1, 1280, 720));
      expired.observe(candidate(2, frame, 1, 1280, 720));
      expired.sample(1, narrow, frame);
      expired.sample(2, broad, frame);
      if (frame >= 10)
        require(expired.choose(frame, 1).id == (frame == 24 ? 2u : 1u), "expired evidence must discard pending relative wins before rebuilding confirmation");
    }
    std::puts("PASS evolving comparisons: lost advantage, skipped bad sample, destruction, replacement and stale-history resets");
  }

  void test_native_comparable_completeness() {
    auto native = analyze(gradient(.04f, .052f));
    const auto smaller = analyze(exponential_depth());
    // Policy-level completeness jitter isolates coverage ordering; real analyzer
    // quality/noise validation remains covered independently above.
    native.completeness = .99f;
    selection_policy policy;
    for (std::uint64_t frame : {8u, 9u, 10u}) {
      policy.observe(candidate(1, frame, 1));
      policy.observe(candidate(2, frame, 1, 1280, 720));
      policy.sample(1, smaller, frame);
      policy.sample(2, native, frame);
    }
    paired_challenge(policy, 10, candidate(1, 10, 1), smaller, candidate(2, 10, 1, 1280, 720), native);
    for (std::uint64_t frame = 13; frame < 40; ++frame) {
      auto changing_native = native;
      if (frame % 4 < 2) {
        changing_native.histogram_span = smaller.histogram_span;
        changing_native.histogram_populated_bins = smaller.histogram_populated_bins;
        changing_native.histogram_breadth = smaller.histogram_breadth;
      }
      policy.observe(candidate(1, frame, 10000));
      policy.observe(candidate(2, frame, 1, 1280, 720));
      policy.sample(1, smaller, frame);
      policy.sample(2, changing_native, frame);
      require(policy.choose(frame, 2).id == 2, "useful .99-complete native must survive 1.0-complete lower-resolution histogram churn");
    }
    std::puts("PASS native stability: comparable .99/1.0 support, broader lower-resolution source, changing scene histogram");
  }

  void test_static_comparison_cycles() {
    // These policy-level inputs isolate individually plausible tradeoffs. Fixed
    // samples must never produce A->B->C->D->A, even though each adjacent pair
    // would pass the old pairwise histogram/completeness/resolution thresholds.
    const auto base = analyze(gradient());
    const auto make_quality = [&](float completeness, float span, unsigned bins, float breadth) {
      auto quality = base;
      quality.completeness = completeness;
      quality.histogram_span = span;
      quality.histogram_populated_bins = bins;
      quality.histogram_breadth = breadth;
      return quality;
    };
    const auto assert_no_cycle = [](const std::array<depth_quality, 4> &quality, const std::array<unsigned, 4> &scales) {
      // Isolate the ordering property: default fresh-pair hysteresis would make
      // repeated choose calls on these immutable captures a vacuous cycle test.
      policy_options options;
      options.comparison_samples_required = 1;
      selection_policy policy(options);
      for (std::size_t i = 0; i < quality.size(); ++i) {
        policy.observe(candidate(i + 1, 10, 1, scales[i] * 320, scales[i] * 180));
        for (std::uint64_t frame : {8u, 9u, 10u}) policy.sample(i + 1, quality[i], frame);
      }
      for (std::uint64_t starting_id = 1; starting_id <= 4; ++starting_id) {
        std::uint64_t current = starting_id;
        std::array<bool, 5> seen {};
        seen[current] = true;
        for (unsigned repeat = 0; repeat < 100; ++repeat) {
          const auto next = policy.choose(10, current).id;
          require(next >= 1 && next <= 4, "fixed useful comparison must select an existing resource");
          if (next == current) continue;
          require(!seen[next], "frozen content evidence must not revisit a previously replaced source through pairwise tradeoffs");
          seen[next] = true;
          current = next;
        }
      }
    };
    assert_no_cycle({make_quality(1.f, 2, 4, .25f), make_quality(.91f, 3, 6, .375f),
                     make_quality(.82f, 4, 8, .50f), make_quality(.73f, 5, 10, .625f)}, {4, 4, 4, 4});
    // Histogram gains half-resolution B, small narrower-histogram native steps
    // would then lose that gain, allowing the broad scaled source to return.
    assert_no_cycle({make_quality(1.f, 2, 4, .25f), make_quality(1.f, 4, 8, .50f),
                     make_quality(1.f, 3.4f, 7, .425f), make_quality(1.f, 2.8f, 6, .35f)}, {4, 2, 3, 4});
    // The earlier completeness/native staircase must also obey the shared order.
    assert_no_cycle({make_quality(1.f, 2, 4, .25f), make_quality(.81f, 2, 4, .25f),
                     make_quality(.62f, 2, 4, .25f), make_quality(1.f, 2, 4, .25f)}, {2, 3, 4, 2});
    std::puts("PASS fixed-candidate stability: histogram/support cycles, mixed histogram/resolution cycles, native completeness staircase");
  }

  // One source for the Present-coverage driver: its allocation, its sampled
  // contents, which Presents it draws and the capture set it belongs to.
  struct present_source {
    std::uint64_t id = 0;
    unsigned width = 640, height = 360;
    depth_quality quality;
    unsigned cadence = 1, phase = 0;  // Draws on frames where frame % cadence == phase.
    std::uint64_t set = 0;            // Same-geometry capture set; 0 = itself alone.
    std::uint64_t first = 0, last = std::numeric_limits<std::uint64_t>::max(), miss = 0;
    // Its depth arrives by a copy or resolve from a non-candidate (an MSAA
    // scene, for example): current-copy work, but no draw of its own.
    bool copied = false;
    // Every this-many frames (phase 1) its routed copy fails although it had
    // work: a capture fault, not a gap in its cadence.
    unsigned fault_every = 0;
    bool draws(std::uint64_t frame) const {
      return frame >= first && frame <= last && frame != miss && frame % cadence == phase;
    }
  };

  // Replays depth-active Presents as the add-on reports them: every source
  // with work is observed, gets a fresh capture every fourth frame it works
  // (read back `latency` depth-active Presents later, as the add-on's sampler
  // completes it), the policy chooses, then the routed set reports this
  // Present's coverage. A
  // routed set shows a ready current copy exactly when one of its members had
  // work and none of them faulted. flat counts Presents from count_from on
  // whose chosen source showed no current depth; unrendered, promotions on a
  // Present the promoted source itself did not render.
  struct present_run {
    selection_policy policy;
    std::vector<present_source> sources;
    std::uint64_t current = 0, frame = 0, count_from = std::numeric_limits<std::uint64_t>::max();
    unsigned changes = 0, flat = 0, unrendered = 0, latency = 0;
    const char *reason = "";
    struct readback { std::uint64_t id, frame; depth_quality quality; };
    std::vector<readback> readbacks;

    template <class Member>
    bool any_member(std::uint64_t id, Member &&member) const {
      std::uint64_t set = id;
      for (const auto &source : sources)
        if (source.id == id && source.set) set = source.set;
      for (const auto &source : sources)
        if ((source.id == id || (source.set && source.set == set)) && member(source)) return true;
      return false;
    }
    selection_policy::set_presence presence(std::uint64_t id, std::uint64_t at) const {
      return {any_member(id, [at](const present_source &s) { return s.draws(at); }),
              any_member(id, [at](const present_source &s) { return s.draws(at) && !s.copied; })};
    }

    void run(std::uint64_t until) {
      for (; frame < until; ++frame) {
        for (const auto &source : sources) {
          if (!source.draws(frame)) continue;
          policy.observe(candidate(source.id, frame, 1, source.width, source.height));
          if (frame / source.cadence % 4 == 0) readbacks.push_back({source.id, frame, source.quality});
        }
        for (auto it = readbacks.begin(); it != readbacks.end();)
          if (it->frame + latency <= frame) { policy.sample(it->id, it->quality, it->frame); it = readbacks.erase(it); }
          else ++it;
        const auto choice = policy.choose(frame, current);
        if (choice.id != current) {
          ++changes; reason = choice.reason;
          // A promotion on a Present its source did not render has no current
          // copy of the new anchor and no hold of the old one: mono.
          if (current && std::string(choice.reason).rfind("sustained", 0) == 0 &&
              std::none_of(sources.begin(), sources.end(), [&](const auto &s) { return s.id == choice.id && s.draws(frame); }))
            ++unrendered;
        }
        current = choice.id;
        const auto anchor = presence(current, frame);
        const bool faulted = any_member(current, [this](const present_source &s) {
          return s.draws(frame) && s.fault_every && frame % s.fault_every == 1;
        });
        const bool shown = anchor.active && !faulted;
        if (frame >= count_from && !shown) ++flat;
        policy.observe_present(frame, current, shown, anchor, [&](std::uint64_t id) { return presence(id, frame); });
      }
    }
  };

  void test_present_coverage_outranks_resolution() {
    const auto useful = quality_score(.95f);
    for (unsigned cadence : {2u, 3u}) {
      // (1) Interleaved native depth must not make an every-Present source's
      // Presents flat merely because it is larger.
      present_run every;
      every.sources = {{1, 640, 360, useful}, {2, 1280, 720, useful, cadence, 0}};
      every.run(40);
      require(every.current == 1 && every.changes == 1, "every-Present source must be chosen first");
      every.run(400);
      require(every.current == 1 && every.changes == 1, "a larger source drawing only some Presents displaced an every-Present source");

      // (2) An interleaved native incumbent leaves Presents flat that a
      // comparable every-Present source covers: the latter reclaims.
      present_run reclaim;
      reclaim.sources = {{2, 1280, 720, useful, cadence, 0}};
      reclaim.run(40);
      require(reclaim.current == 2, "interleaved native depth must qualify on its own");
      reclaim.sources.push_back({1, 640, 360, useful, 1, 0, 0, 40});
      // The reclaim takes the challenger's three positive captures and three
      // paired comparisons, each waiting for a fresh incumbent capture (one
      // every fourth frame it draws): six incumbent capture periods. Until
      // then the incumbent's skipped Presents have no current depth (the
      // add-on holds the last frame there); none may follow the reclaim.
      reclaim.count_from = 40;
      const std::uint64_t bound = 6 * 4 * cadence;
      while (reclaim.current != 1 && reclaim.frame < 40 + bound) reclaim.run(reclaim.frame + 1);
      require(reclaim.current == 1 && std::string(reclaim.reason) == "sustained comparable depth on more Presents",
              "comparable every-Present depth did not reclaim Presents an interleaved incumbent left flat within six capture periods");
      require(reclaim.flat <= bound * (cadence - 1) / cadence, "the reclaim left more Presents without current depth than its comparisons need");
      const auto flat_at_reclaim = reclaim.flat;
      reclaim.run(400);
      require(reclaim.flat == flat_at_reclaim, "a Present without current depth followed the reclaim");
      reclaim.run(800);
      require(reclaim.current == 1 && reclaim.changes == 2, "Present-coverage reclaim oscillated");
    }

    // (3) Both draw every Present: native resolution still promotes.
    present_run both;
    both.sources = {{1, 640, 360, useful}, {2, 1280, 720, useful, 1, 0, 0, 60}};
    both.run(400);
    require(both.current == 2 && both.changes == 2 && std::string(both.reason) == "sustained comparable depth with better source coverage",
            "native depth drawing every Present lost its resolution preference");

    // (4) The same native lifetime first draws every other Present, then every
    // Present: its earlier misses predate fresh comparisons and stop blocking.
    present_run steadies;
    steadies.sources = {{1, 640, 360, useful}, {2, 1280, 720, useful, 2, 0, 0, 0, 299}, {2, 1280, 720, useful, 1, 0, 0, 300}};
    steadies.run(300);
    require(steadies.current == 1, "interleaved native depth promoted before it covered every Present");
    steadies.run(700);
    require(steadies.current == 2 && steadies.changes == 2, "native depth did not promote after it began drawing every Present");
    std::puts("PASS Present coverage: every-Present depth holds and reclaims against 1-in-2/1-in-3 native depth, native promotes once it draws every Present");
  }

  void test_present_coverage_capture_sets() {
    const auto useful = quality_score(.95f);
    // (5) An A/B native incumbent covers every Present as a set: the
    // every-Present lower-resolution source has nothing to reclaim.
    present_run held;
    held.sources = {{2, 1280, 720, useful, 2, 0, 9}, {3, 1280, 720, useful, 2, 1, 9}};
    held.run(40);
    require(held.current == 2 || held.current == 3, "alternating native allocations must qualify");
    held.sources.push_back({1, 640, 360, useful, 1, 0, 0, 40});
    const auto native = held.current;
    held.run(600);
    require(held.current == native && held.changes == 1, "lower-resolution depth displaced an A/B native set that covers every Present");

    // (6) A/B, A/B/C and A/B/C/D native challengers are judged by their whole
    // set and still regain native resolution from an every-Present smaller
    // source. They take over on a Present the promoted member rendered (its
    // own current copy; the other members' copies start after it), never on
    // another member's turn, which would have no copy and no hold.
    for (unsigned members : {2u, 3u, 4u})
      for (unsigned turn = 0; turn < members; ++turn)
      for (unsigned latency = 0; latency < members; ++latency) {
        present_run sets;
        sets.latency = latency;
        sets.sources = {{1, 640, 360, useful}};
        sets.run(40);
        for (unsigned i = 0; i < members; ++i)
          sets.sources.push_back({2 + i, 1280, 720, useful, members, (i + turn) % members, 9, 40});
        sets.run(800);
        require(sets.current >= 2 && sets.current < 2 + members && sets.changes == 2,
                "alternating native allocations were judged one member at a time");
        require(sets.unrendered == 0, "an alternating native set took over on a Present its promoted member did not render");
      }
    std::puts("PASS Present coverage by capture set: A/B native incumbent holds; A/B, A/B/C and A/B/C/D native challengers "
              "promote on the promoted member's own Present");
  }

  void test_present_coverage_hitches() {
    const auto useful = quality_score(.95f);
    // (7) One missed incumbent Present supplies at most one paired win.
    present_run hitch;
    hitch.sources = {{2, 1280, 720, useful}};
    hitch.run(40);
    hitch.sources.push_back({1, 640, 360, useful, 1, 0, 0, 40});
    hitch.sources[0].miss = 200;
    hitch.run(800);
    require(hitch.current == 2 && hitch.changes == 1, "a single missed native Present handed selection to lower resolution");

    // (8) One missed challenger Present resets a running promotion; it
    // promotes after fresh comparisons without misses.
    present_run late;
    late.sources = {{1, 640, 360, useful}, {2, 1280, 720, useful, 1, 0, 0, 60}};
    late.sources[1].miss = 70;
    late.run(80);
    require(late.current == 1, "a challenger with a missed Present won its running comparison");
    late.run(600);
    require(late.current == 2 && late.changes == 2, "one old miss permanently blocked native promotion");

    // (9) Completeness class still comes first: a fuller source drawing every
    // other Present keeps replacing a less complete every-Present source.
    auto partial = useful;
    partial.completeness = .5f;
    present_run fuller;
    fuller.sources = {{1, 1280, 720, partial}, {2, 1280, 720, useful, 2, 0, 0, 40}};
    fuller.run(600);
    require(fuller.current == 2 && fuller.changes == 2 && std::string(fuller.reason) == "sustained fuller nonendpoint depth support",
            "Present coverage overruled materially fuller scene support");
    require(fuller.unrendered == 0, "an every-other-Present challenger took over on a Present it did not render");
    std::puts("PASS Present coverage evidence: one incumbent or challenger miss cannot flip selection; completeness class still ranks first");
  }

  void test_present_coverage_presence() {
    const auto useful = quality_score(.95f);
    // (10) Native depth that reaches its allocation by a copy or resolve every
    // Present (from an MSAA scene, which is no candidate itself) is current on
    // every Present: it is not blocked as missing and promotes on resolution.
    present_run copied;
    copied.sources = {{1, 640, 360, useful}, {2, 1280, 720, useful, 1, 0, 0, 60}};
    copied.sources[1].copied = true;
    copied.run(800);
    require(copied.current == 2 && copied.changes == 2 && std::string(copied.reason) == "sustained comparable depth with better source coverage",
            "native depth fed by a copy every Present was treated as missing Presents");

    // (11) Capture faults count for neither side: an incumbent whose copy
    // fails every other Present although it drew is not a source with gaps,
    // so an every-Present lower-resolution source does not reclaim it.
    present_run faults;
    faults.sources = {{2, 1280, 720, useful}};
    faults.run(40);
    faults.sources[0].fault_every = 2;
    faults.sources.push_back({1, 640, 360, useful, 1, 0, 0, 40});
    faults.run(800);
    require(faults.current == 2 && faults.changes == 1, "capture faults on the incumbent were counted as Presents it left without depth");
    std::puts("PASS Present coverage presence: copy-fed native depth promotes; capture faults count for neither side");
  }

  void test_recovery_coverage() {
    const auto useful = quality_score(.95f);
    // (12) Both sources stop for ten depth-active Presents (a menu that draws
    // only other depth). On resume, recovery picks the source drawn on every
    // Present over the larger one drawn on every other, so no resumed Present
    // is flat while a reclaim runs.
    present_run pause;
    pause.sources = {{1, 640, 360, useful, 1, 0, 0, 0, 399}, {2, 1280, 720, useful, 2, 0, 0, 0, 399},
                     {1, 640, 360, useful, 1, 0, 0, 410}, {2, 1280, 720, useful, 2, 0, 0, 410}};
    pause.run(400);
    require(pause.current == 1, "every-Present depth must be chosen before the pause");
    pause.count_from = 410;
    pause.run(1200);
    require(pause.current == 1 && pause.flat == 0, "recovery picked depth drawn on only some Presents, leaving resumed Presents flat");

    // (13) Materially fuller depth drawn every other Present still beats less
    // complete every-Present depth in recovery: Present coverage passes over a
    // source with gaps only for an unmarked one of the same or a higher class.
    auto partial = useful;
    partial.completeness = .5f;
    present_run fuller;
    fuller.sources = {{1, 1280, 720, partial, 1, 0, 0, 0, 399}, {2, 1280, 720, useful, 2, 0, 0, 0, 399},
                      {1, 1280, 720, partial, 1, 0, 0, 410}, {2, 1280, 720, useful, 2, 0, 0, 410}};
    fuller.run(400);
    require(fuller.current == 2, "fuller depth must win before the pause");
    fuller.run(412);
    require(fuller.current == 2, "recovery put Present coverage above the completeness class");

    // (14) Without coverage gaps the completeness class decides nothing in
    // recovery: native depth .79 complete (class 4) and half-resolution depth
    // .81 complete (class 5), both drawn on every Present, are ranked by the
    // content score as an initial pick always was, so a .02 difference within
    // sampling noise never halves the depth resolution. Whichever source
    // anchors first (lifetime order), recovery after a pause picks native.
    auto native_quality = useful, half_quality = useful;
    native_quality.completeness = .79f;
    half_quality.completeness = .81f;
    for (const bool native_first : {true, false}) {
      const std::uint64_t native_id = native_first ? 1 : 2, half_id = native_first ? 2 : 1;
      present_run boundary;
      boundary.sources = {{native_id, 1280, 720, native_quality, 1, 0, 0, 0, 399}, {half_id, 640, 360, half_quality, 1, 0, 0, 0, 399},
                          {native_id, 1280, 720, native_quality, 1, 0, 0, 410}, {half_id, 640, 360, half_quality, 1, 0, 0, 410}};
      boundary.run(400);
      boundary.run(1200);
      require(boundary.current == native_id, "recovery without coverage gaps let the completeness class override the content score");
    }

    // (15) One order for every lifetime assignment. Native depth drawn on
    // every other Present (.81, class 5, gapped) is passed over: an unmarked
    // source of its class (half resolution, .81, class 5) drew on every
    // Present. The score then ranks the rest: native every-Present depth (.79)
    // ahead of the half-resolution source. A pairwise rule that let the class
    // decide only between differing marks would cycle here (gapped native over
    // the lower class, which beats half resolution on the score, which beats
    // the gapped source on coverage) and pick by lifetime order.
    const std::array<std::array<std::uint64_t, 3>, 6> orders {{{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}}};
    for (const auto &ids : orders) {
      const std::uint64_t gapped_native = ids[0], native_id = ids[1], half_id = ids[2];
      present_run three;
      three.sources = {{gapped_native, 1280, 720, half_quality, 2, 0, 0, 0, 399}, {native_id, 1280, 720, native_quality, 1, 0, 0, 0, 399},
                       {half_id, 640, 360, half_quality, 1, 0, 0, 0, 399},
                       {gapped_native, 1280, 720, half_quality, 2, 0, 0, 410}, {native_id, 1280, 720, native_quality, 1, 0, 0, 410},
                       {half_id, 640, 360, half_quality, 1, 0, 0, 410}};
      three.run(400);
      three.run(411);
      require(three.current == native_id, "recovery depended on lifetime order or ranked a source with coverage gaps over its class");
    }
    std::puts("PASS recovery: every-Present depth wins within a completeness class; fuller depth with gaps still wins across classes; "
              "without coverage gaps the content score ranks, in one order");
  }
}  // namespace

int main() {
  try {
    test_content();
    test_shapes();
    test_workload_and_hysteresis();
    test_flat_menu_and_unknown();
    test_lifetime_and_fairness();
    test_late_native_probe_priority();
    test_native_probe_fairness();
    test_unusable_native_probe_fallback();
    test_native_probe_activity_and_coverage();
    test_native_incumbent_probe_order();
    test_staleness_and_order();
    test_clear_only_activity();
    test_qualified_native_challenge();
    test_equal_size_and_quality_blips();
    test_native_is_not_validity();
    test_comparison_capture_freshness();
    test_completeness_measurement_and_noise();
    test_full_scene_beats_people();
    test_aspect_grid_candidate_quality();
    test_completeness_hysteresis();
    test_sky_support();
    test_histogram_measurement();
    test_histogram_challenge_and_clawback();
    test_histogram_content_guards();
    test_histogram_history_and_lifetimes();
    test_fresh_relative_wins();
    test_evolving_advantage_and_resets();
    test_native_comparable_completeness();
    test_static_comparison_cycles();
    test_present_coverage_outranks_resolution();
    test_present_coverage_capture_sets();
    test_present_coverage_hitches();
    test_present_coverage_presence();
    test_recovery_coverage();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL depth selection: %s\n", error.what());
    return 1;
  }
}
