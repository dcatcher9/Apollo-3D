// SPDX-License-Identifier: GPL-3.0-only
#include "camera_sample_binding.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "camera_sample_observation.h"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
  namespace binding = sunshine_camera_binding;
  namespace scene = sunshine_camera_scene;
  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  binding::submission fixture() {
    binding::submission s;
    s.selection.scope = {0x11, 1, 0x22, 2};
    s.selection.original = {0x33, 3, 7, 2048, 1152, 64, 36, 1920, 1080};
    s.selection.sampled = {0x44, 4};
    s.selection.sample_format = 39; // Opaque native typeless depth format value.
    s.selection.sample_width = 2048;
    s.selection.sample_height = 1152;
    s.selection.crop = {64, 36, 1920, 1080};
    s.selection.layout_epoch = 5;
    s.camera.unit_epoch = 6;
    s.camera.frame = {80, 800};
    s.camera.source = s.selection.original;
    s.camera.A = 0;
    s.camera.B = 0.1;
    s.camera.proof_admitted = true; // Transport must refuse to promote this.
    s.copy = {8, 9, 0x55, 10, 11, 12};
    s.capture_frame = 81; // Independent ReShade capture frame, not SL frame80.
    s.capture_ms = 1000;
    s.projection_associated = true;
    return s;
  }
  struct completion_fixture {
    std::array<float, scene::grid_width * scene::grid_height> pixels;
    binding::readback result;
    explicit completion_fixture(const binding::submission &s, std::uint64_t capture_id) {
      for (std::size_t i = 0; i != pixels.size(); ++i) pixels[i] = static_cast<float>(i + 1) / (pixels.size() + 1);
      result.capture_id = capture_id;
      result.scope = s.selection.scope;
      result.sampled = s.selection.sampled;
      result.source_format = s.selection.sample_format;
      result.source_width = s.selection.sample_width;
      result.source_height = s.selection.sample_height;
      result.crop = s.selection.crop;
      result.values = pixels.data();
      result.value_count = pixels.size();
      result.valid = true;
    }
  };
  std::uint64_t start(binding::mailbox &m, const binding::submission &s, std::uint64_t now = 1000) {
    std::uint64_t id = 0;
    require(m.begin(s, now, id) == binding::status::submitted && id != 0 && m.busy(), "Mailbox did not admit one valid pending submission");
    return id;
  }
  void clean_rejection(const binding::associated_sample &out) {
    require(!out.association_available && !out.projection_associated && !out.sample.metadata.proof_admitted &&
      out.sample.id == 0 && out.sample.capture_ms == 0 && out.sample.raw[0] == 0,
      "Failed completion retained a previous association or pixels");
  }

  void immutable_submission_and_policy_boundary() {
    binding::mailbox m;
    auto s = fixture();
    const auto expected = s;
    const auto id = start(m, s);
    completion_fixture complete(expected, id);
    // Emulate frames passing and newer camera/projection/copy metadata replacing
    // the caller's live state before asynchronous GPU readback completes.
    s.camera.A = 1; s.camera.B = -0.001;
    s.camera.frame = {90, 900};
    s.copy.copy_id = 900;
    s.copy.source_content_version = 901;
    s.capture_frame = 91;
    s.capture_ms = 1100;
    require(m.invalidate_if_changed(s.selection) == binding::status::unchanged, "Later camera/copy incorrectly canceled same-layout sampling");
    binding::associated_sample out;
    require(m.complete(complete.result, 1200, out) == binding::status::associated, "Correct delayed completion was rejected");
    require(out.association_available && out.projection_associated && !m.busy(), "Completed metadata association was not exposed");
    require(out.sample.id == id && out.sample.capture_ms == 1000 && out.captured.capture_frame == 81,
      "Arrival time or latest capture frame replaced submission identity");
    require(out.sample.metadata.A == expected.camera.A && out.sample.metadata.B == expected.camera.B &&
      out.sample.metadata.frame == expected.camera.frame && out.sample.readback_frame == expected.camera.frame &&
      out.sample.readback_source == expected.camera.source, "Completion used the latest mutable camera/source");
    require(out.captured.copy.copy_id == 8 && out.captured.copy.source_content_version == 11 &&
      out.captured.copy.recording_generation == 10 && out.captured.selection == expected.selection,
      "Submission-time copy/scope/layout metadata was mutated");
    require(!out.captured.camera.proof_admitted && !out.sample.metadata.proof_admitted,
      "Metadata association promoted final-color/native-coverage proof");
    for (std::size_t i = 0; i != complete.pixels.size(); ++i)
      require(out.sample.raw[i] == complete.pixels[i], "Raw sample changed during transport");
    complete.pixels[0] = 99;
    require(out.sample.raw[0] != 99, "Mailbox retained borrowed readback storage");
    scene::policy numerical;
    numerical.reset(expected.camera.unit_epoch, 1000);
    require(numerical.observe(out.sample, 1200) == scene::status::proof_missing,
      "Weak bound metadata entered the numerical policy");
    require(m.complete(complete.result, 1201, out) == binding::status::no_pending, "Duplicate completion was accepted");
    clean_rejection(out);
  }

  void busy_cancel_and_recreation() {
    auto s = fixture();
    binding::mailbox m;
    const auto old = start(m, s);
    completion_fixture stale(s, old);
    std::uint64_t attempted = 999;
    require(m.begin(s, 1001, attempted) == binding::status::busy && attempted == 0,
      "Busy begin replaced pending metadata or returned a usable ID");
    m.cancel();
    binding::associated_sample out;
    require(m.complete(stale.result, 1002, out) == binding::status::no_pending, "Canceled request completed");
    clean_rejection(out);
    const auto current = start(m, s, 1003);
    require(current > old, "Cancel reused a readback ID");
    require(m.complete(stale.result, 1004, out) == binding::status::wrong_capture_id && m.busy(),
      "Late old completion consumed a newer pending request");
    completion_fixture correct(s, current);
    require(m.complete(correct.result, 1005, out) == binding::status::associated, "Late stale completion poisoned valid newer request");
    binding::mailbox recreated;
    const auto next = start(recreated, s, 1006);
    require(next > current, "Mailbox recreation reused an old request ID");
    require(recreated.complete(correct.result, 1007, out) == binding::status::wrong_capture_id && recreated.busy(),
      "Completion from destroyed mailbox matched new lifetime");
    recreated.cancel();
    static_assert(!std::is_copy_constructible_v<binding::mailbox> && !std::is_copy_assignable_v<binding::mailbox>,
      "Copying a pending mailbox would permit duplicate completion");
  }

  template<typename Change>
  void completion_mismatch(Change change, binding::status expected) {
    binding::mailbox m;
    const auto s = fixture();
    completion_fixture c(s, start(m, s));
    change(c.result);
    binding::associated_sample out;
    out.association_available = true;
    out.sample.raw[0] = 42;
    require(m.complete(c.result, 1100, out) == expected, "Completion mismatch returned the wrong rejection");
    require(!m.busy(), "Matching invalid completion was not terminal");
    clean_rejection(out);
    require(m.complete(c.result, 1101, out) == binding::status::no_pending, "Failed matching completion could be retried as fresh");
  }
  void sampler_identity_and_layout() {
    completion_mismatch([](auto &r) { ++r.scope.runtime_epoch; }, binding::status::scope_mismatch);
    completion_mismatch([](auto &r) { ++r.scope.device_epoch; }, binding::status::scope_mismatch);
    completion_mismatch([](auto &r) { ++r.sampled.native; }, binding::status::source_mismatch);
    completion_mismatch([](auto &r) { ++r.sampled.lifetime; }, binding::status::source_mismatch);
    completion_mismatch([](auto &r) { ++r.source_format; }, binding::status::format_mismatch);
    completion_mismatch([](auto &r) { ++r.source_width; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { ++r.source_height; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { --r.crop.x; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { --r.crop.y; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { --r.crop.width; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { --r.crop.height; }, binding::status::layout_mismatch);
    // Sampler's documented fallback from invalid crop to full resource must not
    // silently associate a different region with the requested projection.
    completion_mismatch([](auto &r) { r.crop = {0, 0, r.source_width, r.source_height}; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { r.width = 16; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { r.height = 9; }, binding::status::layout_mismatch);
    completion_mismatch([](auto &r) { r.valid = false; }, binding::status::invalid_gpu_result);
    completion_mismatch([](auto &r) { r.values = nullptr; }, binding::status::invalid_gpu_result);
    completion_mismatch([](auto &r) { --r.value_count; }, binding::status::invalid_gpu_result);
    completion_mismatch([](auto &r) { ++r.value_count; }, binding::status::invalid_gpu_result);
  }

  template<typename Change>
  void selection_change(Change change) {
    binding::mailbox m;
    auto s = fixture();
    completion_fixture c(s, start(m, s));
    auto current = s.selection;
    change(current);
    require(m.invalidate_if_changed(current) == binding::status::selection_changed && !m.busy(),
      "Selection change did not cancel old pending binding");
    binding::associated_sample out;
    require(m.complete(c.result, 1100, out) == binding::status::no_pending, "Old completion survived scope/layout invalidation");
    clean_rejection(out);
  }
  void owner_lifetime_invalidation() {
    selection_change([](auto &s) { ++s.scope.runtime; });
    selection_change([](auto &s) { ++s.scope.runtime_epoch; });
    selection_change([](auto &s) { ++s.scope.device; });
    selection_change([](auto &s) { ++s.scope.device_epoch; });
    selection_change([](auto &s) { ++s.original.native; });
    selection_change([](auto &s) { ++s.original.lifetime; });
    selection_change([](auto &s) { ++s.original.viewport; });
    selection_change([](auto &s) { ++s.original.left; });
    selection_change([](auto &s) { ++s.original.extent_width; });
    selection_change([](auto &s) { ++s.sampled.native; });
    selection_change([](auto &s) { ++s.sampled.lifetime; });
    selection_change([](auto &s) { ++s.sample_format; });
    selection_change([](auto &s) { ++s.sample_width; });
    selection_change([](auto &s) { ++s.crop.x; });
    selection_change([](auto &s) { ++s.layout_epoch; });
  }

  void age_and_invalid_submission() {
    const auto s = fixture();
    binding::mailbox m;
    std::uint64_t id = 99;
    require(m.begin(s, 999, id) == binding::status::invalid_submission && !id, "Future capture time was admitted");
    require(m.begin(s, 2500, id) == binding::status::expired && !id, "Stale submission was admitted at exact age boundary");
    completion_fixture c(s, start(m, s, 2400));
    binding::associated_sample out;
    require(m.complete(c.result, 2499, out) == binding::status::associated && out.sample.capture_ms == 1000,
      "A valid delayed completion reset capture age or expired early");
    c.result.capture_id = start(m, s, 2400);
    require(m.complete(c.result, 2500, out) == binding::status::expired, "Arrival time refreshed an old sample");
    clean_rejection(out);
    c.result.capture_id = start(m, s, 1200);
    require(m.complete(c.result, 1199, out) == binding::status::clock_went_backwards, "Completion time before submission accepted");

    const auto reject = [&](binding::submission invalid) {
      id = 99;
      require(m.begin(invalid, 1200, id) == binding::status::invalid_submission && !id && !m.busy(),
        "Malformed scope/source/request was admitted");
    };
    auto bad = s; bad.selection.scope.runtime_epoch = 0; reject(bad);
    bad = s; bad.selection.original.lifetime = 0; reject(bad);
    bad = s; bad.selection.sampled.lifetime = 0; reject(bad);
    bad = s; bad.selection.sample_format = 0; reject(bad);
    bad = s; bad.selection.crop.width = 0; reject(bad);
    bad = s; bad.selection.crop.x = std::numeric_limits<std::uint32_t>::max(); reject(bad);
    bad = s; bad.selection.crop.width = std::numeric_limits<std::uint32_t>::max(); reject(bad);
    bad = s; ++bad.camera.source.lifetime; reject(bad);
    bad = s; ++bad.camera.source.viewport; reject(bad);
    bad = s; bad.camera.unit_epoch = 0; reject(bad);
  }

  void absent_projection_and_no_numerical_policy() {
    auto s = fixture();
    s.projection_associated = false;
    s.camera = {};
    binding::mailbox m;
    completion_fixture c(s, start(m, s));
    // Association is about which readback belongs to which request. It does
    // not fabricate a camera, filter clear depth, or perform depth statistics.
    c.pixels.fill(0.0f);
    c.pixels[0] = std::numeric_limits<float>::quiet_NaN();
    binding::associated_sample out;
    require(m.complete(c.result, 1100, out) == binding::status::associated, "Metadata transport performed numerical calibration");
    require(out.association_available && !out.projection_associated && !out.sample.metadata.proof_admitted &&
      out.sample.metadata.unit_epoch == 0 && std::isnan(out.sample.raw[0]) && out.sample.raw[1] == 0,
      "Absent projection or raw sample values were fabricated");
  }

  struct observation_fixture {
    binding::submission submitted = fixture();
    sunshine_streamline::selected_depth selected;
    sunshine_streamline::evaluation_snapshot evidence;
    observation_fixture() {
      namespace sl = sunshine_streamline;
      const auto &s = submitted.selection;
      const auto &r = s.original;
      selected.resource = r.native; selected.source_id = r.lifetime; selected.layout_epoch = s.layout_epoch;
      selected.frame_index = submitted.capture_frame;
      selected.width = r.width; selected.height = r.height; selected.x = r.left; selected.y = r.top;
      selected.active_width = r.extent_width; selected.active_height = r.extent_height;
      selected.backup_resource = s.sampled.native; selected.backup_id = s.sampled.lifetime; selected.ready = true;
      selected.command_queue = 0x77;
      selected.depth_copy.source.resource = {r.native, r.lifetime};
      selected.depth_copy.backup = {s.sampled.native, s.sampled.lifetime};
      selected.depth_copy.copy_id = submitted.copy.copy_id;
      selected.depth_copy.ledger_epoch = submitted.copy.ledger_epoch;
      selected.depth_copy.recording = {submitted.copy.recording_command, 5, submitted.copy.recording_generation, 7, 8, 9};
      selected.depth_copy.source.version = submitted.copy.source_content_version;
      selected.depth_copy.backup_registration = submitted.copy.backup_registration;
      selected.capture_marker = selected.depth_copy.recording;
      evidence.selection = selected;
      evidence.status = sl::evidence_status::source_associated_evaluation;
      evidence.projection.status = sl::validation_status::valid;
      evidence.projection.depth_offset = -.001; evidence.projection.depth_scale = .1001;
      evidence.decoded = sl::decode_status::ok;
      evidence.successful_evaluation = evidence.frame_correlated = true;
      evidence.epoch = 11; evidence.sequence = 91; evidence.camera_sequence = 90;
      evidence.tick = 999; evidence.camera_tick = 998; evidence.loss_revision = 4;
      evidence.frame = {sl::frame_identity_kind::v1_numeric, 0, 700, 0, true};
      evidence.viewport = 7;
      evidence.tags[0].present = evidence.tags[0].supported = true;
      evidence.tags[0].value.native_resource = r.native;
      evidence.tags[0].value.viewport = evidence.viewport;
    }
    void capture() { binding::capture_camera_observation(submitted, selected, evidence); }
  };

  void captured_probe_is_immutable_and_weak() {
    namespace sl = sunshine_streamline;
    for (const auto level : {binding::observation_level::source, binding::observation_level::command, binding::observation_level::content}) {
      observation_fixture f;
      f.evidence.status = level == binding::observation_level::source ? sl::evidence_status::source_associated_evaluation :
        level == binding::observation_level::command ? sl::evidence_status::command_associated_evaluation : sl::evidence_status::tracked_content_evaluation;
      f.capture();
      require(f.submitted.observation.level == level && f.submitted.observation.binding == binding::observation_binding_status::captured &&
        f.submitted.projection_associated && !f.submitted.camera.proof_admitted, "Production capture lost evidence level or promoted proof");
      require(f.submitted.camera.frame.frame == 80 && f.submitted.capture_frame == 81 && f.submitted.observation.frame_numeric == 700,
        "Production capture conflated sampler, capture and Streamline frame identities");
      binding::mailbox m;
      const auto original = f.submitted;
      completion_fixture complete(original, start(m, original));
      f.evidence.projection.depth_offset = 1; f.evidence.projection.depth_scale = -.003;
      f.evidence.sequence = 999; f.evidence.frame.numeric = 999; f.evidence.selection.depth_copy.copy_id = 900;
      f.capture(); // New bad evidence must not rewrite the mailbox's old envelope.
      binding::associated_sample out;
      require(m.complete(complete.result, 1200, out) == binding::status::associated, "Observation transport rejected a correct delayed readback");
      require(out.captured.camera.A == -.001 && out.captured.camera.B == .1001 &&
        out.captured.observation.sequence == 91 && out.captured.observation.frame_numeric == 700 && out.captured.observation.level == level &&
        !out.captured.camera.proof_admitted && !out.sample.metadata.proof_admitted,
        "Completion retrofitted newer camera/frame/evidence or admitted geometry");
    }
    observation_fixture token;
    token.evidence.frame = {sl::frame_identity_kind::v2_observed_token, 123, 0, 0xabc, false};
    token.capture();
    require(token.submitted.projection_associated && token.submitted.observation.frame_generation == 123 &&
      token.submitted.observation.frame_token == 0xabc && !token.submitted.observation.frame_has_numeric && token.submitted.camera.frame.frame == 80,
      "Observed opaque token fabricated a numerical ReShade frame");
  }

  void probe_capture_rejects_mismatched_envelopes() {
    namespace sl = sunshine_streamline;
    using result = binding::observation_binding_status;
    const auto reject = [](auto mutate, result expected) {
      observation_fixture f;
      f.capture(); // Begin with valid old metadata to test that rejection clears it.
      mutate(f);
      f.capture();
      require(!f.submitted.projection_associated && !f.submitted.camera.proof_admitted &&
        f.submitted.camera.A == 0 && f.submitted.camera.B == 0 && f.submitted.observation.level == binding::observation_level::unavailable &&
        f.submitted.observation.binding == expected, "Production observation capture accepted a mismatch or retained stale camera data");
    };
    reject([](auto &f) { ++f.selected.source_id; }, result::selection_mismatch);
    reject([](auto &f) { ++f.selected.backup_id; }, result::selection_mismatch);
    reject([](auto &f) { ++f.selected.frame_index; }, result::selection_mismatch);
    reject([](auto &f) { ++f.selected.x; }, result::selection_mismatch);
    reject([](auto &f) { ++f.submitted.selection.crop.x; }, result::selection_mismatch);
    reject([](auto &f) { ++f.submitted.camera.source.lifetime; }, result::selection_mismatch);
    reject([](auto &f) { ++f.evidence.selection.resource; }, result::selection_mismatch);
    reject([](auto &f) { ++f.evidence.selection.backup_id; }, result::selection_mismatch);
    reject([](auto &f) { ++f.evidence.selection.depth_copy.copy_id; }, result::copy_mismatch);
    reject([](auto &f) { ++f.evidence.selection.depth_copy.source.resource.lifetime; }, result::copy_mismatch);
    reject([](auto &f) { ++f.evidence.selection.depth_copy.backup.lifetime; }, result::copy_mismatch);
    reject([](auto &f) { ++f.evidence.selection.capture_marker.event; }, result::copy_mismatch);
    reject([](auto &f) { ++f.selected.depth_copy.source.resource.lifetime; }, result::copy_mismatch);
    reject([](auto &f) { f.evidence.successful_evaluation = false; }, result::rejected_camera);
    reject([](auto &f) { f.evidence.projection.depth_scale = 0; }, result::rejected_camera);
    reject([](auto &f) { f.evidence.projection.depth_offset = std::numeric_limits<double>::infinity(); }, result::rejected_camera);
    reject([](auto &f) { f.evidence.decoded = sl::decode_status::unknown_version; }, result::rejected_camera);
    reject([](auto &f) { f.evidence.tags[0].value.type = 49; }, result::rejected_camera);
    reject([](auto &f) { f.evidence.frame.has_numeric = false; }, result::rejected_frame);
    reject([](auto &f) { f.evidence.frame = {sl::frame_identity_kind::v2_observed_token, 0, 0, 0xabc, false}; }, result::rejected_frame);
    reject([](auto &f) { f.evidence.status = sl::evidence_status::observation_lost; }, result::unavailable);
    reject([](auto &f) { f.evidence.status = sl::evidence_status::source_mismatch; }, result::unavailable);
  }
}

int main() {
  try {
    immutable_submission_and_policy_boundary();
    busy_cancel_and_recreation();
    sampler_identity_and_layout();
    owner_lifetime_invalidation();
    age_and_invalid_submission();
    absent_projection_and_no_numerical_policy();
    captured_probe_is_immutable_and_weak();
    probe_capture_rejects_mismatched_envelopes();
    std::puts("PASS camera sample binding: immutable camera/copy/frame, independent request IDs, scope/layout/lifetimes, cancellation, aging and proof boundary");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL camera sample binding: %s\n", e.what());
    return 1;
  }
}
