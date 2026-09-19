// SPDX-License-Identifier: GPL-3.0-only
#include "provided_raw_scene.h"
#include <cstdio>
#include <cstdlib>

static void require(bool value, const char *message) {
  if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static void same_millisecond_reset_rejects_rewrapped_provider_readback(
    sunshine_scene_depth::provider_kind provider) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize same-millisecond reset fixture");
  sunshine_scene_depth::frame frame;
  frame.provider = provider;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  raw.fill(.25f);
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) {
    frame.sequence = i + 1;
    frame.tick = 1000 + i * 250;
    const auto current = sunshine_provided_raw::selected(frame, 1, true);
    const auto measured = sunshine_provided_raw::measured(frame, i + 1, 1, raw);
    output = policy.update(current, &measured, frame.tick);
  }
  require(output.ready && output.H == 4.f, "Same-millisecond reset fixture failed initial calibration");

  // latest_center can be published earlier within the millisecond in which the
  // UI action is handled. The adapter gives it the caller's NEW basis epoch,
  // even though its pixels and capture metadata still predate that action.
  frame.sequence = 5;
  frame.tick = 2000;
  require(policy.reset(2, frame.tick), "Explicit reference reset was refused");
  policy.scene_cut(frame.tick); // Same strict post-action floor as exporter.cpp.
  const auto current = sunshine_provided_raw::selected(frame, 2, true);
  const auto rewrapped = sunshine_provided_raw::measured(frame, 1000000, 2, raw);
  output = policy.update(current, &rewrapped, frame.tick);
  require(!output.ready && !output.calibrated && output.calibration_samples == 0,
    "Same-millisecond old provider pixels seeded the new reference after epoch rewrapping");

  raw.fill(.125f);
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
  require(output.H == 8.f && output.t0 == .125f,
    "Post-reset reference mixed old and new provider observations");
}

static void history_reset_preserves_the_current_shader_domain(
    sunshine_scene_depth::provider_kind provider, float target, bool supported) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize unsupported-zero fixture");
  sunshine_scene_depth::frame frame;
  frame.provider = provider;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  std::uint64_t id = 0;
  const auto capture = [&](float depth, std::uint64_t time) {
    raw.fill(depth);
    frame.tick = time;
    ++frame.sequence;
    const auto current = sunshine_provided_raw::selected(frame, 1, true);
    const auto measured = sunshine_provided_raw::measured(frame, ++id, 1, raw);
    return policy.update(current, &measured, time);
  };
  sunshine_raw_scene::output output;
  for (unsigned i = 0; i < 4; ++i) output = capture(.0001f, 1000 + i * 250);
  require(output.ready, "Representable initial zero did not become ready");
  // A scene reference near 1e-8 is ordinary finite R32 displacement. Only an
  // actual overflowing reciprocal (the tiny-reference case) must remain mono.
  for (unsigned i = 0; i < (supported ? 200u : 1200u) && output.ready; ++i)
    output = capture(target, frame.tick + 50);
  require(output.ready == supported && output.calibrated &&
      (supported ? std::isfinite(output.H) && output.H > 9e7f :
       !std::isfinite(output.H) && output.reason == sunshine_raw_scene::status::unsupported_shader_domain),
    "Reference fixture does not exercise the finite-versus-overflowing FP32 domain");
  const auto held = output;
  ++frame.feedback.revision;
  frame.feedback.reset = true;
  output = capture(target, frame.tick + 50);
  require(output.ready == supported &&
      (supported || output.reason == sunshine_raw_scene::status::unsupported_shader_domain) &&
      output.H == held.H && output.t0 == held.t0,
    "SDK history reset changed the held zero/scale domain");
  frame.feedback.reset = false;
  for (unsigned i = 0; i < 80 && !output.ready; ++i)
    output = capture(.25f, frame.tick + 50);
  require(output.ready && std::abs(output.H * output.t0 - 1.f) <= 2e-7f,
    "Fresh provider evidence failed to recover an unsupported zero without explicit reset");
}

static void provider_switch_restarts_sequence_domain(sunshine_scene_depth::provider_kind first,
    sunshine_scene_depth::provider_kind next) {
  sunshine_raw_scene::policy policy;
  require(policy.reset(1, 1000), "Failed to initialize provider-switch policy");
  sunshine_scene_depth::frame frame;
  frame.provider = first;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  raw.fill(0.25f);
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
  raw.fill(0.125f);
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
  require(output.ready && output.t0 <= calibrated.t0 &&
      std::abs(output.H * output.t0 - 1.f) <= 2e-7f && output.target_H == 8.f,
    "Recovered source failed to pair its current zero with reciprocal scale");
  for (unsigned i = 0; i < 8; ++i) {
    frame.tick += 250; ++frame.sequence;
    current = sunshine_provided_raw::selected(frame, 1, true);
    const auto next = sunshine_provided_raw::measured(frame, 6+i, 1, raw);
    const float previous_zero = output.t0;
    output = policy.update(current, &next, frame.tick);
    require(output.ready && std::abs(output.H * output.t0 - 1.f) <= 2e-7f &&
        output.target_H == 8.f && output.t0 < previous_zero && output.t0 > .125f,
      "Recovered provider failed to update the zero and reciprocal scale together");
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
  sunshine_scene_depth::frame frame;
  frame.provider = sunshine_scene_depth::provider_kind::ngx;
  frame.epoch = 3;
  frame.source_id = 9;
  frame.projection.reversed = true;
  frame.projection.direction_supplied = true;
  std::array<float, 32 * 18> raw;
  raw.fill(0.25f);
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
  require(output.ready && std::abs(output.H - 4.f) < 1e-5, "Rotating NGX textures restarted calibration");
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
  require(output.ready && output.H == initial, "Temporary capture gap moved the current zero or reciprocal scale");
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
  require(output.ready && std::abs(output.H - 4.f / 3.f) < 1e-5,
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
  require(output.ready && std::abs(output.H - 4.f) < 1e-5,
    "A newly declared direction failed to start fresh calibration");
  sl.projection.direction_supplied = false; sl.projection.supplied = true;
  require(sunshine_provided_raw::selected(sl, 1, true).direction == sunshine_depth::depth_orientation::reversed,
    "Validated projection no longer implies a known direction");
  frame.source_id = 0;
  require(sunshine_provided_raw::selected(frame, 1, true).source.lifetime == 0,
    "Unknown NGX feature generation inherited Streamline's epoch fallback");
  std::puts("Provided raw scene tests passed");
}
