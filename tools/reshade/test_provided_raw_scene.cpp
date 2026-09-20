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
