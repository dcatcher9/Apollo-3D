// SPDX-License-Identifier: GPL-3.0-only
#include "depth_selection_policy.h"

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

  void test_manual_recovery_watch() {
    manual_recovery_watch watch;
    require(watch.inactive_frame() == 0, "a new manual watch must have no inactive interval");
    require(!watch.update(7, true, 10, 1000), "a rendered manual choice must remain pinned");
    require(watch.inactive_frame() == 0, "rendered manual depth cannot start recovery evidence");
    require(!watch.update(7, false, 11, 1100), "the first missed render must begin a grace interval");
    require(watch.inactive_frame() == 11, "replacement evidence must follow the first inactive observation");
    require(!watch.update(7, false, 12, 3099), "1999 milliseconds of inactivity must retain a manual choice");
    require(watch.update(7, false, 13, 3100), "2000 milliseconds of continuous inactivity must allow recovery checks");
    require(watch.inactive_frame() == 11, "expired grace must retain its original evidence boundary");
    require(watch.update(7, false, 14, 3500), "continued inactivity must not restart an expired grace interval");
    require(!watch.update(7, true, 15, 3501), "the original buffer rendering again must cancel recovery");
    require(watch.inactive_frame() == 0, "resumed manual depth must clear the old evidence boundary");

    // A buffer used on alternating presents is healthy. Separate short gaps may
    // not accumulate into a reload timeout, irrespective of the frame rate.
    for (std::uint64_t round = 0; round < 8; ++round) {
      const auto frame = 20 + 2 * round, now = 4000 + 2000 * round;
      require(!watch.update(7, false, frame, now), "each new short gap must start a fresh grace interval");
      require(!watch.update(7, true, frame + 1, now + 1999), "intermittently rendered manual depth must stay pinned");
      require(watch.inactive_frame() == 0, "a rendered frame must prevent banking earlier inactive time");
    }

    require(!watch.update(7, false, 40, 20000), "an inactive manual source must begin a new interval");
    require(!watch.update(8, false, 41, 21999), "a newly chosen manual identity cannot inherit the old timeout");
    require(watch.inactive_frame() == 41, "a new manual lifetime must establish its own evidence boundary");
    require(!watch.update(8, false, 42, 23998), "replacement manual identity must receive the full grace interval");
    require(watch.update(8, false, 43, 23999), "a new manual identity may recover only after its own grace expires");
    require(!watch.update(0, false, 44, 24000), "automatic mode has no manual timeout to expire");
    require(watch.inactive_frame() == 0, "clearing manual selection must discard its inactive interval");
    require(!watch.update(8, false, 45, 27000), "reselecting a previous identity cannot resurrect its expired timeout");

    manual_recovery_watch rollback;
    require(!rollback.update(9, false, 50, 10000), "clock rollback scenario must begin inactive");
    require(!rollback.update(9, false, 51, 9999), "a backward clock must not underflow into immediate recovery");
    require(rollback.inactive_frame() == 51, "clock rollback must restart the evidence boundary with its timer");
    require(!rollback.update(9, false, 52, 11998), "clock rollback must preserve a complete new grace interval");
    require(rollback.update(9, false, 53, 11999), "recovery must resume after the restarted grace interval");
    std::puts("PASS manual inactivity: active pins, intermittent presents, exact two-second grace, identity changes, reset and clock rollback");
  }

  void test_manual_recovery_evidence() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    const auto flat = analyze(std::vector<float>(width * height, 1.f));
    policy.observe(candidate(2, 100, 1, 1280, 720));
    for (std::uint64_t frame : {98u, 99u, 100u}) policy.sample(2, useful, frame);
    require(policy.evidence(2)->confirmed_good, "replacement scenario must begin with historical useful evidence");
    require(!policy.confirmed_since(2, 100, 100), "pre-reload evidence and a capture exactly on the boundary cannot release a manual pin");
    policy.observe(candidate(2, 101, 1, 1280, 720));
    policy.sample(2, useful, 101);
    require(!policy.confirmed_since(2, 101, 100), "one post-reload positive capture cannot borrow historical confirmation");
    policy.sample(2, useful, 101);
    require(!policy.confirmed_since(2, 101, 100), "duplicate post-reload captures cannot manufacture replacement confirmation");
    policy.sample(2, useful, 99);
    require(!policy.confirmed_since(2, 101, 100), "late historical readback cannot advance recovery evidence");
    policy.observe(candidate(2, 102, 1, 1280, 720));
    policy.sample(2, useful, 102);
    require(policy.confirmed_since(2, 102, 100), "two independent useful captures after inactivity must qualify a currently rendered replacement");
    require(!policy.confirmed_since(2, 103, 100), "ordinary activity grace cannot make a nonrendering replacement safe for manual recovery");
    policy.observe(candidate(2, 103, 1, 1280, 720));
    require(policy.confirmed_since(2, 103, 100), "a currently rendered replacement may use its recent completed readbacks");
    policy.sample(2, flat, 103);
    require(policy.evidence(2)->confirmed_good, "ordinary negative hysteresis must retain historical confirmation for this scenario");
    require(!policy.confirmed_since(2, 103, 100), "a latest flat capture must prevent recovery even before ordinary bad-content hysteresis expires");
    require(!policy.confirmed_since(0, 103, 100) && !policy.confirmed_since(999, 103, 100), "missing replacement identities cannot provide recovery evidence");

    policy.erase(2);
    policy.sample(2, useful, 104);
    require(!policy.confirmed_since(2, 104, 100), "late readbacks must not resurrect a destroyed replacement");
    policy.observe(candidate(20, 104, 1, 1280, 720)); // Same native pointer, new lifetime.
    policy.sample(2, useful, 105);
    require(!policy.confirmed_since(20, 104, 100), "a reused resource address must not inherit the destroyed lifetime's evidence");
    policy.sample(20, useful, 104);
    require(!policy.confirmed_since(20, 104, 100), "a replacement lifetime must independently collect two positive captures");
    policy.observe(candidate(20, 105, 1, 1280, 720));
    policy.sample(20, useful, 105);
    require(policy.confirmed_since(20, 105, 100), "a new lifetime may qualify once its own post-reload captures are confirmed");
    std::puts("PASS manual recovery evidence: strict post-inactivity boundary, independent positive captures, live rendering, bad latest contents and resource lifetimes");
  }

  void test_clear_only_activity() {
    selection_policy policy;
    const auto useful = analyze(gradient());
    policy.observe(candidate(1, 100, 1));
    policy.sample(1, useful, 99);
    policy.sample(1, useful, 100);
    require(policy.choose(100).id == 1, "rendered scene must qualify before a clear-only frame");
    require(policy.confirmed_since(1, 100, 98), "rendered source must initially provide fresh recovery evidence");

    // A pinned resource is observed even when the caller's automatic loop skips
    // it. Cached positive captures cannot turn its later clear into scene work.
    policy.observe(candidate(1, 101, 0));
    require(policy.choose(101, 1).id == 0, "clear-only current source cannot remain an active automatic candidate");
    require(policy.next_probe(101) == 0, "clear-only source cannot schedule a content capture");
    require(!policy.confirmed_since(1, 101, 98), "clear-only activity cannot release another manual pin using cached proof");
    policy.sample(1, useful, 101); // Late preserved pixels still cannot establish current rendering.
    require(!policy.confirmed_since(1, 101, 98), "new readback cannot substitute for a current scene draw");

    policy.observe(candidate(2, 101, 1));
    policy.sample(2, useful, 100);
    policy.sample(2, useful, 101);
    require(policy.choose(101, 1).id == 2, "current useful replacement must outrank a clear-only pinned resource");
    require(policy.confirmed_since(2, 101, 99), "replacement scene must retain ordinary fresh recovery eligibility");
    policy.observe(candidate(1, 102, 1));
    require(policy.choose(102, 1).id == 1, "resumed rendering must restore the retained scene without inventing a new lifetime");
    require(policy.confirmed_since(1, 102, 99), "resumed rendering may use its fresh positive samples");
    std::puts("PASS clear-only activity: no ready candidate/probe/recovery proof, fresh replacement and resumed rendering");
  }

  void test_manual_recovery_evidence_freshness() {
    policy_options options;
    options.evidence_stale_frames = 5;
    selection_policy policy(options);
    const auto useful = analyze(gradient());
    policy.observe(candidate(1, 102, 1));
    policy.sample(1, useful, 101);
    policy.sample(1, useful, 102);
    require(policy.confirmed_since(1, 102, 100), "fresh post-inactivity evidence should qualify initially");
    require(!policy.confirmed_since(1, 101, 100), "a reversed frame clock cannot qualify evidence from a future rendered frame");
    policy.observe(candidate(1, 108, 1));
    require(!policy.confirmed_since(1, 108, 100), "rendering again cannot revive expired content confidence without fresh readbacks");
    policy.sample(1, useful, 108);
    require(!policy.confirmed_since(1, 108, 100), "one capture after evidence expiration cannot revive the old positive count");
    policy.observe(candidate(1, 109, 1));
    policy.sample(1, useful, 109);
    require(policy.confirmed_since(1, 109, 100), "two fresh captures must restore recovery eligibility after evidence expires");

    options.good_samples_required = 3;
    selection_policy three_samples(options);
    three_samples.observe(candidate(3, 201, 1));
    three_samples.sample(3, useful, 199);
    three_samples.sample(3, useful, 200);
    three_samples.sample(3, useful, 201);
    three_samples.observe(candidate(3, 202, 1));
    three_samples.sample(3, useful, 202);
    require(!three_samples.confirmed_since(3, 202, 200), "manual recovery must honor a configured three-capture requirement after the boundary");
    three_samples.observe(candidate(3, 203, 1));
    three_samples.sample(3, useful, 203);
    require(three_samples.confirmed_since(3, 203, 200), "three new positive captures must satisfy the configured recovery requirement");
    std::puts("PASS manual recovery freshness: expired evidence, frame ordering and configured confirmation count");
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
    capture(2, broad, 14);
    capture(2, broad, 13); // An out-of-order completion must not count.
    require(policy.choose(14, 1).id == 1, "a second challenger sample cannot reuse the incumbent capture");
    capture(1, narrow, 15);
    require(policy.choose(15, 1).id == 2, "three distinct paired wins must eventually promote a better same-size source");
    require(policy.choose(15, 2).id == 2 && policy.pending_challenger_id() == 0,
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
    test_manual_recovery_watch();
    test_manual_recovery_evidence();
    test_manual_recovery_evidence_freshness();
    test_clear_only_activity();
    test_qualified_native_challenge();
    test_equal_size_and_quality_blips();
    test_native_is_not_validity();
    test_comparison_capture_freshness();
    test_completeness_measurement_and_noise();
    test_full_scene_beats_people();
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
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL depth selection: %s\n", error.what());
    return 1;
  }
}
