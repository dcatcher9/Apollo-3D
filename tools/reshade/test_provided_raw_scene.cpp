// SPDX-License-Identifier: GPL-3.0-only
#include "provided_raw_scene.h"
#include <cstdio>
#include <cstdlib>

static void require(bool value, const char *message) {
  if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static constexpr double test_normalization = .5;
static constexpr sunshine_scene_gain::limits test_budget{
  .5, double(sunshine_camera_scene::reference_zpd) * .5,
  double(sunshine_camera_scene::reference_zpd) * .5, test_normalization};
// depth_range has a known maximum of 1.5 * center, independent of its mean.
static double expected_gain(float center) { return test_normalization / double(1.5f * center); }
static bool gain_matches(float actual, float center) {
  return std::abs(double(actual) - expected_gain(center)) <= expected_gain(center) * 2e-7;
}
static void depth_range(std::array<float, 32 * 18> &raw, float center) {
  for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = center * (i % 2 ? 1.5f : .5f);
}

static void same_millisecond_reset_rejects_rewrapped_provider_readback(
    sunshine_scene_depth::provider_kind provider) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize same-millisecond reset fixture");
  policy.configure(test_budget);
  sunshine_scene_depth::frame frame;
  frame.provider = provider;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  depth_range(raw, .25f);
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = i + 1;
    frame.tick = 1000 + i * 250;
    const auto current = sunshine_provided_raw::selected(frame, 1, true);
    const auto measured = sunshine_provided_raw::measured(frame, i + 1, 1, raw);
    output = policy.update(current, &measured, frame.tick);
  }
  require(output.ready && gain_matches(output.H, .25f), "Same-millisecond reset fixture failed initial calibration");

  // latest_center can be published earlier within the millisecond in which the
  // diagnostic reset is handled. The adapter gives it the caller's NEW basis epoch,
  // even though its pixels and capture metadata still predate that action.
  frame.sequence = 5;
  frame.tick = 2000;
  require(policy.reset(2, frame.tick), "Explicit reference reset was refused");
  policy.configure(test_budget);
  policy.scene_cut(frame.tick); // Same strict post-action floor as exporter.cpp.
  const auto current = sunshine_provided_raw::selected(frame, 2, true);
  const auto rewrapped = sunshine_provided_raw::measured(frame, 1000000, 2, raw);
  output = policy.update(current, &rewrapped, frame.tick);
  require(!output.ready && !output.calibrated && output.calibration_samples == 0,
    "Same-millisecond old provider pixels seeded the new reference after epoch rewrapping");

  depth_range(raw, .125f);
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = 6 + i;
    frame.tick = 2001 + i * 250;
    const auto fresh_current = sunshine_provided_raw::selected(frame, 2, true);
    // The rejected high ID must not poison order for fresh lower-ID readbacks.
    const auto fresh = sunshine_provided_raw::measured(frame, 5 + i, 2, raw);
    output = policy.update(fresh_current, &fresh, frame.tick);
    require(output.calibration_samples == i + 1 && output.ready == (i == 3),
      "Same-millisecond rejection poisoned the fresh post-reset capture window");
  }
  require(gain_matches(output.H, .125f) && output.t0 == .125f,
    "Post-reset reference mixed old and new provider observations");
}

static void history_reset_preserves_the_current_shader_domain(
    sunshine_scene_depth::provider_kind provider, float target, bool supported) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize extreme-range fixture");
  policy.configure(test_budget);
  sunshine_scene_depth::frame frame;
  frame.provider = provider;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  std::uint64_t id = 0;
  const auto capture = [&](float depth, std::uint64_t time) {
    depth_range(raw, depth);
    frame.tick = time;
    ++frame.sequence;
    const auto current = sunshine_provided_raw::selected(frame, 1, true);
    const auto measured = sunshine_provided_raw::measured(frame, ++id, 1, raw);
    return policy.update(current, &measured, time);
  };
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) output = capture(.0001f, 1000 + i * 250);
  require(output.ready, "Representable initial zero did not become ready");
  // The fixture has Qmax=1.5*target. Check the requested L/Qmax domain,
  // not a magnitude threshold inherited from the former mean normalization.
  const double requested = expected_gain(target);
  require((std::isfinite(requested) && requested <= std::numeric_limits<float>::max()) == supported,
    "Extreme-range fixture does not straddle the FP32 gain domain");
  // An overflowing target invalidates fresh evidence without replacing the
  // established finite gain. Supported adaptation retains its rate limit.
  for (unsigned i = 0; i < (supported ? 400u : 1u) && output.ready; ++i)
    output = capture(target, frame.tick + 50);
  require(output.ready == supported && output.calibrated &&
      std::isfinite(output.H) && (supported ? output.H > requested * .99 && output.H <= requested :
       output.reason == sunshine_raw_scene::status::unsupported_shader_domain),
    "Reference fixture does not exercise the finite-versus-overflowing FP32 domain");
  const auto held = output;
  ++frame.feedback.revision;
  frame.feedback.reset = true;
  output = capture(target, frame.tick + 50);
  require(output.ready && output.H == held.H && output.t0 == held.t0,
    "SDK history reset changed the held zero/scale domain");
  frame.feedback.reset = false;
  // Both gain directions now obey the same rate bound. Allow recovery from
  // ~3e7 to ~1 over several octaves; do not require the old immediate decrease.
  for (unsigned i = 0; i < 1200; ++i) {
    const float previous_gain = output.H;
    output = capture(.25f, frame.tick + 50);
    require(output.ready && output.H >= previous_gain / std::exp2(.05) - std::max(1e-6, double(previous_gain) * 1e-7),
      "Recovery bypassed the gain decrease rate bound");
    if (output.H == output.target_H) break;
  }
  require(output.ready && gain_matches(output.H, .25f) && gain_matches(output.target_H, .25f) &&
      output.t0 >= .125f && output.t0 <= .375f,
    "Fresh provider evidence failed to recover gain and observed-range zero without explicit reset");
}

static void provider_switch_restarts_sequence_domain(sunshine_scene_depth::provider_kind first,
    sunshine_scene_depth::provider_kind next) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize provider-switch policy");
  policy.configure(test_budget);
  sunshine_scene_depth::frame frame;
  frame.provider = first;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  depth_range(raw, .25f);
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = 100000 + i;
    frame.tick = 1000 + i * 250;
    const auto current = sunshine_provided_raw::selected(frame, 1, true);
    output = policy.update(current, nullptr, frame.tick);
    policy.observe(sunshine_provided_raw::measured(frame, 10000 + i, 1, raw), current.frame, frame.tick);
    output = policy.evaluate(current, frame.tick);
  }
  require(output.ready, "Initial provider did not calibrate");
  const auto previous = sunshine_provided_raw::measured(frame, 10004, 1, raw);

  frame.provider = next;
  frame.sequence = 1;
  frame.tick = 2000;
  auto current = sunshine_provided_raw::selected(frame, 1, true);
  require(!(current.source == previous.metadata.source), "Provider identity did not change");
  policy.bind(current, frame.tick);
  policy.observe(previous, current.frame, frame.tick);
  output = policy.evaluate(current, frame.tick);
  require(output.reason == sunshine_raw_scene::status::calibrating && output.calibration_samples == 0,
    "Provider rebind kept the old sequence floor or admitted the old provider's packet");
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = i + 1;
    frame.tick = 2000 + i * 250;
    // Native rotation and dynamic resolution do not change this logical domain.
    frame.resource = {300 + i % 2, 1920 + i, 1080 + i, {0, 0, 1920 + i, 1080 + i}};
    current = sunshine_provided_raw::selected(frame, 1, true);
    policy.bind(current, frame.tick);
    policy.observe(sunshine_provided_raw::measured(frame, i + 1, 1, raw), current.frame, frame.tick);
    output = policy.evaluate(current, frame.tick);
    require(output.calibration_samples == i + 1 && output.ready == (i == 3),
      "New provider with smaller sequence/sample IDs did not get one fresh calibration window");
  }
  const auto calibrated = output;
  auto older = frame;
  --older.sequence;
  older.tick += 250;
  depth_range(raw, .125f);
  const auto old_current = sunshine_provided_raw::selected(older, 1, true);
  policy.bind(old_current, older.tick);
  policy.observe(sunshine_provided_raw::measured(older, 1000000, 1, raw), current.frame, older.tick);
  require(policy.evaluate(old_current, older.tick).reason == sunshine_raw_scene::status::frame_mismatch,
    "Same-provider rebind cleared the current-frame ordering guard");
  output = policy.evaluate(current, older.tick);
  require(output.ready && output.H == calibrated.H && output.t0 == calibrated.t0,
    "Old same-provider packet changed calibration or poisoned current rendering");
  frame.tick += 500;
  ++frame.sequence;
  current = sunshine_provided_raw::selected(frame, 1, true);
  policy.observe(sunshine_provided_raw::measured(frame, 5, 1, raw), current.frame, frame.tick);
  output = policy.evaluate(current, frame.tick);
  require(output.ready && output.t0 < calibrated.t0 && output.t0 > .125f && output.target_t0 == .125 &&
      output.H > calibrated.H && output.H <= calibrated.H * std::exp2(.25) + 1e-6 &&
      output.H*(calibrated.t0-output.t0)/test_normalization <= .25 + 2e-7 && gain_matches(output.target_H, .125f),
    "Recovered source failed bounded independent gain/zero adaptation");
  for (unsigned i = 0; i < 8; ++i) {
    frame.tick += 250; ++frame.sequence;
    current = sunshine_provided_raw::selected(frame, 1, true);
    const auto next = sunshine_provided_raw::measured(frame, 6+i, 1, raw);
    const float previous_zero = output.t0;
    const float previous_gain = output.H;
    output = policy.update(current, &next, frame.tick);
    require(output.ready && gain_matches(output.target_H, .125f) && output.H >= previous_gain &&
        output.H <= previous_gain * std::exp2(.25) + 1e-6 && output.t0 <= previous_zero && output.t0 >= .125f &&
        output.H*(previous_zero-output.t0)/test_normalization <= .25 + 2e-7,
      "Recovered provider exceeded gain or zero movement budget");
  }
  require(output.ready && output.H > calibrated.H && output.t0 < calibrated.t0,
    "Rejected high-ID old packet poisoned subsequent ordered zero-plane refinement");
}

// Retained provided-raw references across frame-generation switches.
using retained_policy = sunshine_provided_raw::retained_policy;
using raw_status = sunshine_raw_scene::status;

struct provided_source {
  sunshine_scene_depth::frame frame;
  std::uint64_t basis_epoch = 1, id = 0;
  std::array<float, 32 * 18> raw{};
  // The next present of this encoding. Its adapter tick may precede the caller's clock.
  sunshine_raw_scene::selected_frame present(std::uint64_t tick, bool ready = true) {
    ++frame.sequence;
    frame.tick = tick;
    return sunshine_provided_raw::selected(frame, basis_epoch, ready);
  }
  // A completed readback of the latest present, captured at capture_ms.
  sunshine_raw_scene::sample packet(float center, std::uint64_t capture_ms) {
    depth_range(raw, center);
    auto captured = frame;
    captured.tick = capture_ms;
    return sunshine_provided_raw::measured(captured, ++id, basis_epoch, raw);
  }
};
static provided_source ngx_source(std::uint64_t feature = 3) {
  provided_source value;
  value.frame.provider = sunshine_scene_depth::provider_kind::ngx;
  value.frame.epoch = 3;
  value.frame.source_id = feature; // NGX feature generation.
  value.frame.sequence = 40843;
  value.frame.feedback.revision = 4;
  value.frame.projection.reversed = value.frame.projection.direction_supplied = true;
  return value;
}
static provided_source sl_source() {
  provided_source value;
  value.frame.provider = sunshine_scene_depth::provider_kind::streamline;
  value.frame.epoch = 4;
  value.frame.viewport = 1;
  value.frame.source_id = (std::uint64_t{1} << 63) | 1; // FG-scoped viewport identity.
  value.frame.sequence = 13247;
  value.frame.projection.reversed = value.frame.projection.direction_supplied = true;
  return value;
}
static void start(retained_policy &policy, std::uint64_t now = 1000) {
  require(policy.reset(1, now), "Failed to initialize retained provided-raw fixture");
  policy.configure(test_budget);
}
static sunshine_raw_scene::output present(retained_policy &policy, provided_source &source,
    std::uint64_t now, bool ready = true) {
  const auto current = source.present(now, ready);
  policy.bind(current, now);
  return policy.evaluate(current, now);
}
// One present whose packet was captured at that present's clock.
static sunshine_raw_scene::output capture(retained_policy &policy, provided_source &source,
    float center, std::uint64_t now) {
  const auto current = source.present(now);
  const auto packet = source.packet(center, now);
  return policy.update(current, &packet, now);
}
static sunshine_raw_scene::output calibrate(retained_policy &policy, provided_source &source,
    float center, std::uint64_t first) {
  sunshine_raw_scene::output out;
  for (unsigned i = 0; i < 4; ++i) out = capture(policy, source, center, first + i * 250);
  require(out.ready && out.calibration_samples == 4 && gain_matches(out.H, center),
    "Retained fixture encoding failed its startup window");
  return out;
}
static bool retained_without_target(const sunshine_raw_scene::output &out, const sunshine_raw_scene::output &reference) {
  return !out.ready && out.calibrated && out.reason == raw_status::no_target &&
    out.calibration_samples == 4 && out.H == reference.H && out.t0 == reference.t0;
}

static void fg_round_trip_retains_reference() {
  retained_policy policy;
  start(policy);
  auto ngx = ngx_source();
  auto sl = sl_source();
  const auto calibrated = calibrate(policy, ngx, .25f, 1000);
  // FG on: the first Streamline present has no depth yet; projection takes the rest.
  auto out = present(policy, sl, 2000, false);
  require(!out.ready && out.reason == raw_status::depth_unavailable && out.calibration_samples == 0,
    "Streamline present without depth did not start its own encoding");

  // FG off 35 s later: NGX returns with a larger sequence and a newer revision.
  ngx.frame.sequence += 9000;
  ngx.frame.feedback.revision = 5;
  const auto during_absence = ngx.packet(.125f, 36000); // Younger than target expiry at return.
  const auto current = ngx.present(36990);
  const auto own_tick = ngx.packet(.125f, 36990); // The re-entry present's own earlier NGX tick.
  const std::uint64_t now = 37000;
  policy.bind(current, now);
  out = policy.evaluate(current, now);
  require(retained_without_target(out, calibrated),
    "Returning NGX lost its reference, rendered without a target or kept another provider's watermark");
  for (const auto *stale : {&during_absence, &own_tick}) {
    policy.observe(*stale, current.frame, now);
    require(retained_without_target(policy.evaluate(current, now), calibrated),
      "A packet captured before re-entry became the returning encoding's target");
  }

  // The first packet captured at the re-entry bind renders the retained values.
  const auto fresh_current = ngx.present(now);
  const auto fresh = ngx.packet(.125f, now);
  out = policy.update(fresh_current, &fresh, now + 20);
  require(out.ready && out.reason == raw_status::ready && out.calibration_samples == 4 &&
      out.H == calibrated.H && out.t0 == calibrated.t0 && gain_matches(float(out.target_H), .125f),
    "Fresh post-re-entry target restarted calibration or credited the absence");
  out = capture(policy, ngx, .125f, now + 120);
  require(out.ready && out.H > calibrated.H && out.H <= calibrated.H * std::exp2(.1) * (1. + 1e-6),
    "Retained gain did not resume bounded adaptation after re-entry");
}

static void reentry_requires_fresh_target() {
  retained_policy policy;
  start(policy);
  auto ngx = ngx_source();
  auto sl = sl_source();
  const auto calibrated = calibrate(policy, ngx, .25f, 1000);
  present(policy, sl, 2000, false);
  ngx.frame.sequence += 9000;
  require(retained_without_target(present(policy, ngx, 37000), calibrated), "Re-entry fixture did not suspend rendering");
  // A continuously bound source holds across a revision bump; a returning one has no current evidence.
  ngx.frame.feedback.revision = 5;
  require(retained_without_target(present(policy, ngx, 37010), calibrated),
    "Revision bump after re-entry rendered the retained reference without a fresh target");
  ngx.frame.feedback.reset = true;
  require(retained_without_target(capture(policy, ngx, .25f, 37020), calibrated),
    "Fresh history-reset packet counted as the returning encoding's target");
  ngx.frame.feedback.reset = false;
  auto out = capture(policy, ngx, .25f, 37030);
  require(out.ready && out.reason == raw_status::ready, "First fresh non-reset target did not end re-entry");
  out = present(policy, ngx, 37030 + sunshine_camera_scene::target_expiry_ms);
  require(out.ready && out.reason == raw_status::holding_reference,
    "Ordinary target expiry after re-entry lost the in-basis reference hold");
}

static void projection_absence_requires_fresh_target() {
  for (const bool left : {true, false}) {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    const auto calibrated = calibrate(policy, ngx, .25f, 1000);
    if (left) policy.leave(); // The projection path took the FG presents.
    ngx.frame.sequence += 9000;
    auto out = present(policy, ngx, 37000);
    if (!left) {
      require(out.ready && out.reason == raw_status::holding_reference && out.H == calibrated.H,
        "Rebinding the active encoding lost today's in-basis hold");
      continue;
    }
    require(retained_without_target(out, calibrated), "Return from the projection path rendered a 35 s old target");
    out = capture(policy, ngx, .25f, 37010);
    require(out.ready && out.H == calibrated.H, "Return from the projection path did not accept a fresh target");
  }
}

static void lru_reuses_value_initialized_slots() {
  retained_policy policy;
  start(policy);
  std::array<provided_source, 5> sources;
  std::array<sunshine_raw_scene::output, 4> calibrated;
  for (unsigned i = 0; i < sources.size(); ++i) sources[i] = ngx_source(10 + i);
  for (unsigned i = 0; i < calibrated.size(); ++i)
    calibrated[i] = calibrate(policy, sources[i], .1f * (i + 1), 1000 + i * 2000);

  // A fifth encoding takes the least recently used slot (the first one) as new.
  auto &fifth = sources[4];
  const auto current = fifth.present(9000);
  const auto before = fifth.packet(.5f, 8999);
  policy.bind(current, 9000);
  policy.observe(before, current.frame, 9000);
  auto out = policy.evaluate(current, 9000);
  require(!out.calibrated && out.reason == raw_status::calibrating && out.calibration_samples == 0 && out.H == 0.f,
    "Reused slot inherited the evicted encoding or admitted a packet captured before admission");
  const auto at_admission = fifth.packet(.5f, 9000);
  policy.observe(at_admission, current.frame, 9000);
  out = policy.evaluate(current, 9000);
  require(out.reason == raw_status::calibrating && out.calibration_samples == 1,
    "Reused slot did not start at the pool epoch with its floor at admission");

  for (unsigned i = 1; i < 4; ++i) {
    sources[i].frame.sequence += 100;
    require(retained_without_target(present(policy, sources[i], 9100 + i * 10), calibrated[i]),
      "LRU admission evicted an encoding other than the least recently used one");
  }
  out = present(policy, sources[0], 9200);
  require(!out.calibrated && out.calibration_samples == 0 && out.reason == raw_status::calibrating,
    "Evicted encoding kept its reference");
  for (unsigned i = 1; i < 4; ++i)
    require(retained_without_target(present(policy, sources[i], 9210 + i * 10), calibrated[i]),
      "Readmitting an evicted encoding evicted a more recently used one");
}

static void retained_policy_owns_clock_sources_and_budget() {
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    calibrate(policy, sl, .25f, 1000);
    calibrate(policy, ngx, .25f, 3000);
    const auto current = ngx.present(3700);
    policy.bind(current, 3700);
    auto out = policy.evaluate(current, 3700);
    require(!out.ready && out.reason == raw_status::clock_went_backwards, "Retained policy accepted a backwards clock");
    // The pool clock owns time: an encoding last seen earlier cannot accept it.
    out = present(policy, sl, 2500);
    require(!out.ready && out.reason == raw_status::clock_went_backwards,
      "An inactive encoding accepted time before the pool clock");
    out = present(policy, ngx, 3800);
    require(!out.ready && out.calibrated && out.reason == raw_status::no_target,
      "Clock regression kept the active encoding's target");
  }
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    const auto calibrated = calibrate(policy, ngx, .25f, 1000);
    auto unknown = ngx;
    unknown.frame.source_id = 0; // NGX without a feature generation.
    auto out = present(policy, unknown, 2000);
    require(!out.ready && out.reason == raw_status::invalid_source, "Invalid provided source was not reported as invalid");
    // An unknown identity is not a return: the encoding continues exactly as one policy does.
    sunshine_raw_scene::policy single;
    single.reset(1, 1000);
    single.configure(test_budget);
    auto twin = ngx_source();
    for (unsigned i = 0; i < 4; ++i) {
      const auto current = twin.present(1000 + i * 250);
      const auto packet = twin.packet(.25f, 1000 + i * 250);
      single.update(current, &packet, 1000 + i * 250);
    }
    auto twin_unknown = twin;
    twin_unknown.frame.source_id = 0;
    single.update(twin_unknown.present(2000), nullptr, 2000);
    const auto expected = single.update(twin.present(2010), nullptr, 2010);
    out = present(policy, ngx, 2010);
    require(out.ready == expected.ready && out.reason == expected.reason && out.calibrated &&
        out.H == calibrated.H && out.H == expected.H && out.t0 == expected.t0,
      "An invalid source changed the active encoding's behaviour");
    auto other_epoch = ngx;
    other_epoch.basis_epoch = 2;
    out = present(policy, other_epoch, 2020);
    require(!out.ready && out.reason == raw_status::basis_epoch_mismatch, "Basis epoch mismatch was not reported");
  }
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    calibrate(policy, ngx, .25f, 1000);
    present(policy, sl, 2000, false);
    auto budget = test_budget;
    budget.normalization = test_normalization * .5;
    policy.configure(budget);
    ngx.frame.sequence += 100;
    present(policy, ngx, 2100);
    const auto out = capture(policy, ngx, .25f, 2110);
    const double expected = expected_gain(.25f) * .5;
    require(out.ready && std::abs(out.target_H - expected) <= expected * 2e-7,
      "A budget change did not reach an inactive encoding");
  }
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    calibrate(policy, ngx, .25f, 1000);
    // Fixture Recenter: a new epoch and a cut in the same millisecond. The first
    // encoding admitted afterwards and one admitted later both keep that floor.
    require(policy.reset(2, 5000), "Retained policy refused a newer basis epoch");
    policy.configure(test_budget);
    policy.scene_cut(5000);
    ngx.basis_epoch = sl.basis_epoch = 2;
    for (auto *source : {&ngx, &sl}) {
      const auto current = source->present(5000);
      const auto rewrapped = source->packet(.125f, 5000);
      const auto out = policy.update(current, &rewrapped, 5000);
      require(!out.calibrated && out.calibration_samples == 0,
        "Same-millisecond reset admitted a rewrapped pre-reset packet");
    }
    calibrate(policy, ngx, .125f, 5001);
    calibrate(policy, sl, .125f, 6001);
  }
  {
    // As with one policy, the first encoding after a reset may adopt a packet
    // captured since the reset; an encoding admitted later starts at its bind.
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    for (auto *source : {&ngx, &sl}) {
      const std::uint64_t now = source == &ngx ? 1100 : 1200;
      const auto current = source->present(now);
      const auto queued = source->packet(.25f, now - 50);
      const auto out = policy.update(current, &queued, now);
      require(out.calibration_samples == (source == &ngx ? 1u : 0u),
        "Admission floor did not follow the pool reset for the first encoding and the bind for later ones");
    }
  }
}

static void retained_reference_invalidation() {
  for (const bool new_epoch : {true, false}) {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    calibrate(policy, ngx, .25f, 1000);
    present(policy, sl, 2000, false);
    if (new_epoch) {
      require(policy.reset(2, 3000), "Retained policy refused a newer basis epoch");
      policy.configure(test_budget);
      ngx.basis_epoch = 2;
    } else {
      policy.discard(3000); // Fixture Recenter on the projection path keeps the epoch.
    }
    ngx.frame.sequence += 100;
    const auto current = ngx.present(3000);
    policy.bind(current, 3000);
    auto out = policy.evaluate(current, 3000);
    require(!out.calibrated && out.calibration_samples == 0 && out.reason == raw_status::calibrating,
      "Basis reset or discard kept a retained reference");
    if (!new_epoch) {
      const auto at_discard = ngx.packet(.25f, 3000);
      policy.observe(at_discard, current.frame, 3000);
      require(policy.evaluate(current, 3000).calibration_samples == 0, "Discard admitted a same-millisecond packet");
    }
    calibrate(policy, ngx, .25f, 3001);
  }
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    const auto calibrated = calibrate(policy, ngx, .25f, 1000);
    present(policy, sl, 2000, false);
    policy.scene_cut(37000); // Disable path or explicit cut while NGX is absent.
    ngx.frame.sequence += 9000;
    const auto current = ngx.present(37000);
    const auto at_cut = ngx.packet(.125f, 37000);
    require(retained_without_target(policy.update(current, &at_cut, 37000), calibrated),
      "An absent encoding accepted a target captured at its cut");
    require(capture(policy, ngx, .125f, 37001).ready, "An absent encoding rejected a target after its cut");
  }
  {
    // Feature recreation, layout epoch, viewport and direction are new encodings.
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    calibrate(policy, ngx, .25f, 1000);
    std::uint64_t now = 3000;
    for (unsigned kind = 0; kind < 4; ++kind, now += 10) {
      auto changed = ngx;
      if (kind == 0) ++changed.frame.source_id;
      else if (kind == 1) ++changed.frame.epoch;
      else if (kind == 2) ++changed.frame.viewport;
      else changed.frame.projection.reversed = false;
      const auto current = changed.present(now);
      ngx.frame.sequence = changed.frame.sequence;
      const auto previous = ngx.packet(.25f, now);
      const auto out = policy.update(current, &previous, now);
      require(!out.calibrated && out.calibration_samples == 0,
        "A changed NGX identity reused the previous encoding's reference or packets");
    }
    auto sl = sl_source();
    calibrate(policy, sl, .25f, 4000);
    auto next = sl;
    ++next.frame.epoch;
    const auto current = next.present(5000);
    sl.frame.sequence = next.frame.sequence;
    const auto previous = sl.packet(.25f, 5000);
    const auto out = policy.update(current, &previous, 5000);
    require(!out.calibrated && out.calibration_samples == 0,
      "A new Streamline epoch reused the previous encoding's reference or packets");
  }
  {
    retained_policy policy;
    start(policy);
    auto ngx = ngx_source();
    auto sl = sl_source();
    capture(policy, ngx, .25f, 1000);
    require(capture(policy, ngx, .25f, 1250).calibration_samples == 2, "Partial window fixture did not collect two samples");
    present(policy, sl, 1300, false);
    ngx.frame.sequence += 10;
    const auto out = present(policy, ngx, 1400);
    require(!out.calibrated && out.calibration_samples == 0 && out.reason == raw_status::calibrating,
      "A partial startup window survived the encoding's absence");
    calibrate(policy, ngx, .25f, 1500);
  }
}

static void retained_provider_switch_keeps_reference(bool ngx_first) {
  retained_policy policy;
  start(policy);
  auto first = ngx_first ? ngx_source() : sl_source();
  auto next = ngx_first ? sl_source() : ngx_source();
  const auto first_reference = calibrate(policy, first, .25f, 1000);
  // A brand-new provider starts from zero in its own sequence domain.
  auto out = present(policy, next, 2000);
  require(!out.calibrated && out.calibration_samples == 0 && out.reason == raw_status::calibrating,
    "A new provider inherited another provider's reference");
  const auto next_reference = calibrate(policy, next, .125f, 2000);
  require(retained_without_target(present(policy, first, 3000), first_reference),
    "Switching back lost the first provider's reference");
  out = capture(policy, first, .25f, 3010);
  require(out.ready && out.H == first_reference.H && out.t0 == first_reference.t0,
    "Switching back did not resume the first provider's reference");
  require(retained_without_target(present(policy, next, 3100), next_reference),
    "Switching again lost the second provider's reference");
}

int main() {
  for (const auto provider : {sunshine_scene_depth::provider_kind::streamline, sunshine_scene_depth::provider_kind::ngx}) {
    history_reset_preserves_the_current_shader_domain(provider, 1e-8f, true);
    history_reset_preserves_the_current_shader_domain(provider, 1e-40f, false);
  }
  same_millisecond_reset_rejects_rewrapped_provider_readback(sunshine_scene_depth::provider_kind::streamline);
  same_millisecond_reset_rejects_rewrapped_provider_readback(sunshine_scene_depth::provider_kind::ngx);
  provider_switch_restarts_sequence_domain(sunshine_scene_depth::provider_kind::streamline,
    sunshine_scene_depth::provider_kind::ngx);
  provider_switch_restarts_sequence_domain(sunshine_scene_depth::provider_kind::ngx,
    sunshine_scene_depth::provider_kind::streamline);
  fg_round_trip_retains_reference();
  reentry_requires_fresh_target();
  projection_absence_requires_fresh_target();
  lru_reuses_value_initialized_slots();
  retained_policy_owns_clock_sources_and_budget();
  retained_reference_invalidation();
  retained_provider_switch_keeps_reference(true);
  retained_provider_switch_keeps_reference(false);
  sunshine_raw_scene::policy policy;
  policy.reset(1, 1000);
  policy.configure(test_budget);
  sunshine_scene_depth::frame frame;
  frame.provider = sunshine_scene_depth::provider_kind::ngx;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = true;
  frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  depth_range(raw, .25f);
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = i + 1;
    frame.tick = 1000 + i * 250;
    // Alternate resources and active sizes with one unchanged depth encoding.
    frame.resource = {100 + i % 2, 1920 + i, 1080 + i, {0, 0, 1919 + i, 1079 + i}};
    auto current = sunshine_provided_raw::selected(frame, 1, true);
    policy.bind(current, frame.tick);
    policy.observe(sunshine_provided_raw::measured(frame, i + 1, 1, raw), current.frame, frame.tick);
    output = policy.evaluate(current, frame.tick);
  }
  require(output.ready && gain_matches(output.H, .25f), "Rotating NGX textures restarted calibration");
  const float initial = output.H;
  frame.sequence++;
  frame.tick += 200;
  auto gap = sunshine_provided_raw::selected(frame, 1, false);
  require(!policy.evaluate(gap, frame.tick).ready, "Missing NGX pixels kept stereo ready");
  frame.sequence++;
  frame.tick += 200;
  auto returned = sunshine_provided_raw::selected(frame, 1, true);
  policy.bind(returned, frame.tick);
  output = policy.evaluate(returned, frame.tick);
  require(output.ready && output.H == initial, "Temporary capture gap moved the current stereo gain");
  // New feature generation cannot use the previous feature's completed sample.
  auto old = sunshine_provided_raw::measured(frame, 30, 1, raw);
  frame.source_id++;
  frame.sequence++;
  frame.tick += 200;
  auto recreated = sunshine_provided_raw::selected(frame, 1, true);
  policy.bind(recreated, frame.tick);
  policy.observe(old, recreated.frame, frame.tick);
  require(!policy.evaluate(recreated, frame.tick).ready, "Recreated feature adopted old encoding samples");
  // Explicit recalibration rejects a readback captured before the reset.
  policy.reset(2, frame.tick + 100);
  policy.configure(test_budget);
  recreated = sunshine_provided_raw::selected(frame, 2, true);
  policy.bind(recreated, frame.tick + 100);
  auto stale = sunshine_provided_raw::measured(frame, 31, 2, raw);
  policy.observe(stale, recreated.frame, frame.tick + 100);
  require(policy.evaluate(recreated, frame.tick + 100).calibration_samples == 0,
    "Recalibrate admitted a previously queued sample");

  // Streamline's viewport/observation epoch is the logical encoding owner.
  // Native resource rotation and dynamic allocation size must not reset it.
  sunshine_raw_scene::policy sl_policy;
  sl_policy.reset(1, 5000);
  sl_policy.configure(test_budget);
  sunshine_scene_depth::frame sl;
  sl.provider = sunshine_scene_depth::provider_kind::streamline;
  sl.epoch = 12; sl.viewport = 4;
  sl.projection.direction_supplied = true;
  const auto sl_identity = sunshine_provided_raw::selected(sl, 1, true).source;
  require(sl_identity.lifetime == sl.epoch && sl_identity.viewport == sl.viewport,
    "Streamline without feature id lost its epoch/viewport identity");
  for (unsigned i = 0; i < 4; ++i) {
    sl.sequence = i + 1; sl.tick = 5000 + i * 250;
    sl.resource = {200 + i % 2, 3840 - i * 100, 2160 - i * 50, {i, i, 1920 - i, 1080 - i}};
    const auto current = sunshine_provided_raw::selected(sl, 1, true);
    require(current.source == sl_identity, "Streamline resource rotation changed the logical encoding");
    sl_policy.bind(current, sl.tick);
    sl_policy.observe(sunshine_provided_raw::measured(sl, i + 1, 1, raw), current.frame, sl.tick);
    output = sl_policy.evaluate(current, sl.tick);
  }
  require(output.ready && std::abs(output.H - test_normalization / .875) < 1e-6,
    "Streamline raw fallback failed to calibrate with known normal direction and no projection");
  auto old_sl = sunshine_provided_raw::measured(sl, 20, 1, raw);
  sl.epoch++; sl.sequence++; sl.tick += 250;
  auto new_sl = sunshine_provided_raw::selected(sl, 1, true);
  sl_policy.bind(new_sl, sl.tick);
  sl_policy.observe(old_sl, new_sl.frame, sl.tick);
  require(!sl_policy.evaluate(new_sl, sl.tick).ready,
    "New Streamline observation epoch reused a previous logical encoding sample");
  sl.viewport++;
  require(!(sunshine_provided_raw::selected(sl, 1, true).source == new_sl.source),
    "Distinct Streamline viewports shared a logical encoding");

  // A default or stale reversed bit is not a declaration. Unknown direction
  // must stay mono even when depth pixels and center readbacks are available.
  sunshine_raw_scene::policy unknown_policy;
  unknown_policy.reset(1, 7000);
  unknown_policy.configure(test_budget);
  sl.projection = {}; sl.projection.reversed = true;
  for (unsigned i = 0; i < 4; ++i) {
    sl.sequence++; sl.tick = 7000 + i * 250;
    const auto current = sunshine_provided_raw::selected(sl, 1, true);
    require(current.direction == sunshine_depth::depth_orientation::automatic,
      "Undeclared projection direction became normal or reversed");
    unknown_policy.bind(current, sl.tick);
    unknown_policy.observe(sunshine_provided_raw::measured(sl, i + 1, 1, raw), current.frame, sl.tick);
    output = unknown_policy.evaluate(current, sl.tick);
    require(!output.ready && output.reason == sunshine_raw_scene::status::direction_unknown,
      "Unknown source direction calibrated or enabled stereo");
  }
  sl.projection.direction_supplied = true;
  for (unsigned i = 0; i < 4; ++i) {
    sl.sequence++; sl.tick += 250;
    const auto current = sunshine_provided_raw::selected(sl, 1, true);
    unknown_policy.bind(current, sl.tick);
    unknown_policy.observe(sunshine_provided_raw::measured(sl, i + 5, 1, raw), current.frame, sl.tick);
    output = unknown_policy.evaluate(current, sl.tick);
  }
  require(output.ready && gain_matches(output.H, .25f),
    "A newly declared direction failed to start fresh calibration");
  sl.projection.direction_supplied = false; sl.projection.supplied = true;
  require(sunshine_provided_raw::selected(sl, 1, true).direction == sunshine_depth::depth_orientation::reversed,
    "Validated projection no longer implies a known direction");
  frame.source_id = 0;
  require(sunshine_provided_raw::selected(frame, 1, true).source.lifetime == 0,
    "Unknown NGX feature generation inherited Streamline's epoch fallback");
  std::puts("Provided raw scene tests passed");
}
