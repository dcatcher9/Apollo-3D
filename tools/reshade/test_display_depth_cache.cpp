// SPDX-License-Identifier: GPL-3.0-only
#include "display_depth_cache.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
  using namespace sunshine_streamline;
  using namespace sunshine_streamline::provider;
  using sunshine_depth::frame_depth;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }
  bool equal(const char *a, const char *b) { return std::strcmp(a, b) == 0; }

  frame_depth fresh() {
    frame_depth value;
    auto &p = value.provided;
    p.epoch = 11; p.sequence = 17; p.tick = 1000; p.viewport = 3;
    p.projection = {0.2, 2.5, true, true, true, 0.5, -0.125};
    p.jitter = {0.375f, -0.25f, 1600, 900, true};
    p.resource = {51, 1920, 1080, {13, 19, 1600, 900}, sunshine_scene_depth::resource_kind::display_depth};
    p.native_state = 8; p.proof = sunshine_scene_depth::state_proof::declared;
    p.valid_until = sunshine_scene_depth::lifetime::at_call;
    p.source_id = 23; p.source_frame_generation = 29; p.source_frame_token = 31;
    p.source_frame_numeric = 37; p.source_frame_has_numeric = true; p.source_frame_explicit = true;
    p.frame_generation_input = true; p.observation_revision = 41; p.feedback = {43, false};
    value.projection = {true, 11, 3, 0.2, 2.5, 0.5, -0.125};
    value.source_resource = {51}; value.resource = {53}; value.shader_resource = {59};
    value.width = 1920; value.height = 1080;
    value.x = 13; value.y = 19; value.active_width = 1600; value.active_height = 900;
    value.source_id = 23; value.layout_epoch = 61; value.frame_index = 71; value.runtime_epoch = 73;
    value.aligned_viewport_assumed = true; value.capture_marker = {79, 83, 89, 97, 101, 103};
    value.depth_copy.copy_id = 107; value.depth_copy.backup = {53, 109}; value.backup_id = 109;
    value.command_queue = 113; value.ready = true; value.frame_generation_active = true;
    value.detected_orientation = sunshine_depth::depth_orientation::reversed;
    value.orientation = sunshine_depth::depth_orientation::normal;
    value.orientation_agreeing_frames = 127; value.calibrated = true;
    value.calibration_samples = 131; value.calibration_frame = 137;
    value.raw_anchor = 0.25f; value.raw_gain = 0.75f;
    return value;
  }
  display_cache_context context() { return {3840, 2160, true, 113, 53, 59, 1016}; }
  depth_capture::display_decision hold() {
    depth_capture::display_decision out;
    out.action = depth_capture::display_action::hold;
    out.hold = depth_capture::hold_kind::frame_generation;
    out.reason = "retained_copy";
    return out;
  }
  void arm(display_depth_cache &cache, const frame_depth &value) {
    cache.commit(value, 139, 40, 3840, 2160);
  }

  void same_metadata(const sunshine_scene_depth::frame &a, const sunshine_scene_depth::frame &b) {
    require(a.epoch == b.epoch && a.sequence == b.sequence && a.tick == b.tick && a.viewport == b.viewport &&
      a.provider == b.provider && a.source_id == b.source_id && a.observation_revision == b.observation_revision,
      "hold changed original source identity or timestamp");
    const auto &x = a.projection; const auto &y = b.projection;
    require(x.depth_offset == y.depth_offset && x.depth_scale == y.depth_scale && x.supplied == y.supplied &&
      x.reversed == y.reversed && x.direction_supplied == y.direction_supplied &&
      x.raw_scale == y.raw_scale && x.raw_bias == y.raw_bias, "hold changed source projection/encoding");
    require(a.jitter.x == b.jitter.x && a.jitter.y == b.jitter.y && a.jitter.width == b.jitter.width &&
      a.jitter.height == b.jitter.height && a.jitter.supplied == b.jitter.supplied, "hold changed original jitter");
    require(a.resource.native == b.resource.native && a.resource.width == b.resource.width &&
      a.resource.height == b.resource.height && a.resource.kind == b.resource.kind &&
      a.resource.area.left == b.resource.area.left && a.resource.area.top == b.resource.area.top &&
      a.resource.area.width == b.resource.area.width && a.resource.area.height == b.resource.area.height,
      "hold changed source resource layout");
    require(a.native_state == b.native_state && a.proof == b.proof && a.valid_until == b.valid_until &&
      a.source_frame_generation == b.source_frame_generation && a.source_frame_token == b.source_frame_token &&
      a.source_frame_numeric == b.source_frame_numeric && a.source_frame_has_numeric == b.source_frame_has_numeric &&
      a.source_frame_explicit == b.source_frame_explicit && a.frame_generation_input == b.frame_generation_input &&
      a.feedback.revision == b.feedback.revision && a.feedback.reset == b.feedback.reset,
      "hold changed immutable source evidence");
  }
  void same_image(const frame_depth &a, const frame_depth &b) {
    same_metadata(a.provided, b.provided);
    const auto &x = a.projection; const auto &y = b.projection;
    require(x.supplied == y.supplied && x.epoch == y.epoch && x.viewport == y.viewport &&
      x.A == y.A && x.B == y.B && x.raw_scale == y.raw_scale && x.raw_bias == y.raw_bias,
      "hold changed resolved projection");
    require(a.source_resource == b.source_resource && a.resource == b.resource && a.shader_resource == b.shader_resource &&
      a.width == b.width && a.height == b.height && a.x == b.x && a.y == b.y &&
      a.active_width == b.active_width && a.active_height == b.active_height && a.source_id == b.source_id &&
      a.layout_epoch == b.layout_epoch && a.runtime_epoch == b.runtime_epoch &&
      a.aligned_viewport_assumed == b.aligned_viewport_assumed && a.backup_id == b.backup_id &&
      a.command_queue == b.command_queue && a.ready == b.ready && a.frame_generation_active == b.frame_generation_active,
      "hold changed private copy identity, crop or source policy");
    require(a.capture_marker.command == b.capture_marker.command &&
      a.capture_marker.object_generation == b.capture_marker.object_generation &&
      a.capture_marker.recording_generation == b.capture_marker.recording_generation &&
      a.capture_marker.event == b.capture_marker.event && a.capture_marker.epoch == b.capture_marker.epoch &&
      a.capture_marker.loss == b.capture_marker.loss && a.depth_copy.copy_id == b.depth_copy.copy_id &&
      content::same(a.depth_copy.backup, b.depth_copy.backup), "hold changed copy provenance");
    require(a.detected_orientation == b.detected_orientation && a.orientation == b.orientation &&
      a.orientation_agreeing_frames == b.orientation_agreeing_frames && a.calibrated == b.calibrated &&
      a.calibration_samples == b.calibration_samples && a.calibration_frame == b.calibration_frame &&
      a.raw_anchor == b.raw_anchor && a.raw_gain == b.raw_gain, "hold changed calibration or gain");
  }
  void cleared(const display_depth_cache &cache, const frame_depth &out) {
    require(!cache.depth().ready && !cache.reference().capture_id && !cache.reference().metadata.tick,
      "invalidated cache still has source authority");
    require(!out.ready && !out.resource.handle && !out.shader_resource.handle && !out.provided.tick &&
      !out.frame_index && !out.reused_depth, "rejected hold exposed previous output");
  }

  void lifecycle() {
    display_depth_cache cache;
    const auto value = fresh();
    auto out = value;
    require(!cache.reuse(hold(), context(), 72, out), "empty cache authorized hold");
    cleared(cache, out);
    arm(cache, value);
    auto ref = cache.reference();
    same_metadata(ref.metadata, value.provided);
    require(ref.capture_id == 139 && ref.real_present == 71 && ref.width == 1920 && ref.height == 1080 &&
      ref.format == 40 && ref.area.left == 13 && ref.area.top == 19 && ref.area.width == 1600 && ref.area.height == 900,
      "cache reference lost original present or pixel layout");
    require(equal(cache.reason(), "none"), "fresh commit retained an invalidation reason");
    for (std::uint64_t present = 72; present != 77; ++present) {
      require(cache.reuse(hold(), context(), present, out), "source-authorized FG hold failed");
      same_image(out, value); same_image(cache.depth(), value);
      require(out.reused_depth && out.frame_index == present && !cache.depth().reused_depth &&
        cache.depth().frame_index == 71 && cache.reference().real_present == 71,
        "hold refreshed original presentation or failed to mark current output");
    }
    cache.invalidate("observer_loss");
    require(!cache.reuse(hold(), context(), 77, out), "hold revived explicitly invalidated pixels");
    cleared(cache, out);
    require(equal(cache.reason(), "observer_loss"), "missing cache hid its initiating invalidation reason");
    auto next = value; next.provided.sequence++; next.provided.tick = 1015; next.frame_index = 77;
    arm(cache, next);
    require(cache.reuse(hold(), context(), 78, out), "fresh commit failed to rearm after invalidation");
    same_image(out, next);
    // Reusing the output as a fresh commit must never extend the original age.
    arm(cache, out);
    require(!cache.reuse(hold(), context(), 79, out) && equal(cache.reason(), "invalid_fresh_copy"),
      "held output was accepted as a fresh commit");
    cleared(cache, out);
  }

  void local_rejections() {
    const char *reasons[] = {"color_shape_changed", "color_shape_changed", "consumer_command_changed",
      "consumer_queue_changed", "consumer_queue_changed", "display_storage_changed", "display_storage_changed",
      "display_storage_changed", "display_storage_changed", "retained_depth_expired", "retained_depth_expired",
      "retained_depth_expired"};
    for (unsigned which = 0; which != sizeof(reasons) / sizeof(reasons[0]); ++which) {
      display_depth_cache cache;
      auto value = fresh(); auto local = context(); auto out = value;
      switch (which) {
        case 0: ++local.color_width; break;
        case 1: ++local.color_height; break;
        case 2: local.immediate_commands = false; break;
        case 3: ++local.queue; break;
        case 4: local.queue = 0; break;
        case 5: ++local.texture; break;
        case 6: ++local.view; break;
        case 7: local.texture = 0; break;
        case 8: local.view = 0; break;
        case 9: value.provided.tick = 0; break;
        case 10: local.now = value.provided.tick - 1; break;
        case 11: local.now = value.provided.tick + sunshine_scene_depth::maximum_source_age_ms; break;
      }
      arm(cache, value);
      require(!cache.reuse(hold(), local, 72, out), "invalid local hold was accepted");
      require(equal(cache.reason(), reasons[which]), "local rejection reason changed");
      cleared(cache, out);
      require(!cache.reuse(hold(), context(), 73, out), "corrected context revived failed cache");
      cleared(cache, out);
    }
    display_depth_cache cache;
    const auto value = fresh(); auto local = context(); frame_depth out;
    arm(cache, value);
    local.now = value.provided.tick + sunshine_scene_depth::maximum_source_age_ms - 1;
    require(cache.reuse(hold(), local, 72, out), "last fresh millisecond was rejected");
    same_image(out, value);
    local.now++;
    require(!cache.reuse(hold(), local, 73, out), "successful hold refreshed original timestamp");
    cleared(cache, out);
  }

  void source_actions_and_commit() {
    for (const auto action : {depth_capture::display_action::invalidate, depth_capture::display_action::copy_fresh}) {
      display_depth_cache cache; const auto value = fresh(); auto out = value;
      arm(cache, value);
      auto decision = hold(); decision.action = action; decision.reason = "source_rejected";
      require(!cache.reuse(decision, context(), 72, out) && equal(cache.reason(), "source_rejected"),
        "non-hold source action reused retained pixels");
      cleared(cache, out);
      require(!cache.reuse(hold(), context(), 73, out), "source invalidation allowed later resurrection");
    }
    for (unsigned which = 0; which != 3; ++which) {
      display_depth_cache cache; auto value = fresh(); auto out = value;
      arm(cache, value);
      if (which == 0) value.ready = false;
      if (which == 1) value.reused_depth = true;
      cache.commit(value, which == 2 ? 0 : 139, 40, 3840, 2160);
      require(!cache.reuse(hold(), context(), 72, out) && equal(cache.reason(), "invalid_fresh_copy"),
        "invalid fresh commit left prior cache available");
      cleared(cache, out);
    }
  }
}

int main() {
  try {
    lifecycle();
    local_rejections();
    source_actions_and_commit();
    std::puts("display depth cache tests passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
