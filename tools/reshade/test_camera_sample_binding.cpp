// SPDX-License-Identifier: GPL-3.0-only
#include "camera_sample_binding.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "camera_sample_observation.h"
#include "depth_selection_policy.h"
#include "raw_scene_policy.h"

#include <array>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    std::vector<float> pixels;
    binding::readback result;
    explicit completion_fixture(const binding::submission &s, std::uint64_t capture_id) :
        pixels(std::size_t(s.grid_width) * s.grid_height) {
      for (std::size_t i = 0; i != pixels.size(); ++i) pixels[i] = static_cast<float>(i + 1) / (pixels.size() + 1);
      result.capture_id = capture_id;
      result.scope = s.selection.scope;
      result.sampled = s.selection.sampled;
      result.source_format = s.selection.sample_format;
      result.source_width = s.selection.sample_width;
      result.source_height = s.selection.sample_height;
      result.crop = s.selection.crop;
      result.width = s.grid_width;
      result.height = s.grid_height;
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
      out.sample.id == 0 && out.sample.capture_ms == 0 && out.sample.raw[0] == 0 &&
      out.sample.width == 0 && out.sample.height == 0 && out.sample.count == 0,
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
    // The old numerical fixture accepts only its fixed layout. Explicitly adapt
    // this legacy test case; live dynamic grids never enter that fixed payload.
    scene::sample physical;
    require(out.sample.width == scene::grid_width && out.sample.height == scene::grid_height &&
      out.sample.count == physical.raw.size(), "Legacy fixture silently changed point layout");
    physical.id = out.sample.id;
    physical.capture_ms = out.sample.capture_ms;
    physical.metadata = out.sample.metadata;
    physical.readback_frame = out.sample.readback_frame;
    physical.readback_source = out.sample.readback_source;
    std::copy_n(out.sample.raw.begin(), physical.raw.size(), physical.raw.begin());
    require(numerical.observe(physical, 1200) == scene::status::proof_missing,
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
    bad = s; bad.grid_width = 0; reject(bad);
    bad = s; bad.grid_height = 0; reject(bad);
    bad = s; bad.grid_width = bad.selection.crop.width + 1; reject(bad);
    bad = s; bad.grid_width = sunshine_depth_statistics::maximum_tiles; bad.grid_height = 2; reject(bad);
  }

  void absent_projection_and_no_numerical_policy() {
    auto s = fixture();
    s.projection_associated = false;
    s.camera = {};
    binding::mailbox m;
    completion_fixture c(s, start(m, s));
    // Association is about which readback belongs to which request. It does
    // not fabricate a camera, filter clear depth, or perform depth statistics.
    std::fill(c.pixels.begin(), c.pixels.end(), 0.0f);
    c.pixels[0] = std::numeric_limits<float>::quiet_NaN();
    binding::associated_sample out;
    require(m.complete(c.result, 1100, out) == binding::status::associated, "Metadata transport performed numerical calibration");
    require(out.association_available && !out.projection_associated && !out.sample.metadata.proof_admitted &&
      out.sample.metadata.unit_epoch == 0 && std::isnan(out.sample.raw[0]) && out.sample.raw[1] == 0,
      "Absent projection or raw sample values were fabricated");
  }

  binding::submission dynamic_fixture(std::uint32_t width, std::uint32_t height) {
    auto s = fixture();
    s.selection.original.width = s.selection.original.extent_width = width;
    s.selection.original.height = s.selection.original.extent_height = height;
    s.selection.original.left = s.selection.original.top = 0;
    s.selection.sample_width = width;
    s.selection.sample_height = height;
    s.selection.crop = {0, 0, width, height};
    s.camera.source = s.selection.original;
    const auto grid = sunshine_depth_statistics::tile_grid(width, height);
    s.grid_width = grid.x;
    s.grid_height = grid.y;
    return s;
  }

  void dynamic_grid_and_crop_are_frozen() {
    for (const auto extent : {binding::rectangle{0, 0, 3440, 1440}, binding::rectangle{0, 0, 1080, 1920},
        binding::rectangle{0, 0, 1024, 1024}, binding::rectangle{0, 0, 3, 2}}) {
      binding::mailbox m;
      auto submitted = dynamic_fixture(extent.width, extent.height);
      completion_fixture c(submitted, start(m, submitted));
      const auto expected_width = submitted.grid_width, expected_height = submitted.grid_height;
      submitted.grid_width = submitted.grid_height = 1; // New caller state is unrelated.
      binding::associated_sample out;
      require(m.complete(c.result, 1100, out) == binding::status::associated,
        "Valid aspect-dependent point grid was rejected");
      require(out.sample.width == expected_width && out.sample.height == expected_height &&
        out.sample.count == c.pixels.size() && out.captured.grid_width == expected_width &&
        out.captured.grid_height == expected_height, "Dynamic grid dimensions were not frozen with the capture");
      for (std::size_t i = 0; i != c.pixels.size(); ++i)
        require(out.sample.raw[i] == c.pixels[i], "Dynamic grid was truncated, resampled or reordered");
      c.pixels.back() = 99;
      require(out.sample.raw[out.sample.count - 1] != 99, "Dynamic grid retained borrowed storage");
    }

    binding::mailbox m;
    const auto old = dynamic_fixture(3440, 1440);
    completion_fixture stale(old, start(m, old));
    auto resized = dynamic_fixture(1080, 1920);
    require(m.invalidate_if_changed(resized.selection) == binding::status::selection_changed,
      "Aspect and source-size change retained an old pending grid");
    completion_fixture current(resized, start(m, resized));
    binding::associated_sample out;
    require(m.complete(stale.result, 1100, out) == binding::status::wrong_capture_id && m.busy(),
      "Old grid completion consumed the resized pending request");
    require(m.complete(current.result, 1100, out) == binding::status::associated,
      "Resized pending request was lost after stale completion");

    auto cropped = old;
    cropped.selection.crop = {0, 0, 1920, 1080};
    const auto cropped_grid = sunshine_depth_statistics::tile_grid(1920, 1080);
    cropped.grid_width = cropped_grid.x;
    cropped.grid_height = cropped_grid.y;
    completion_fixture before_crop(old, start(m, old));
    require(m.invalidate_if_changed(cropped.selection) == binding::status::selection_changed,
      "Active crop change retained a pending grid from the full allocation");
    completion_fixture after_crop(cropped, start(m, cropped));
    require(m.complete(before_crop.result, 1100, out) == binding::status::wrong_capture_id && m.busy(),
      "Old crop completion consumed the current request");
    // Same element count does not establish the same point locations.
    std::swap(after_crop.result.width, after_crop.result.height);
    require(m.complete(after_crop.result, 1100, out) == binding::status::layout_mismatch,
      "Transposed point layout was accepted because its element count matched");
    clean_rejection(out);
  }

  void sparse_selector_points_do_not_veto_full_image_geometry() {
    namespace raw = sunshine_raw_scene;
    for (const auto direction : {raw::orientation::normal, raw::orientation::reversed}) {
      auto submitted = dynamic_fixture(192, 108);
      std::vector<float> q(192 * 108, 0.f);
      // Real foreground lies between tile centers. Its full-image moments are
      // valid even though every selector point observes clear background.
      q.front() = .25f;
      q.back() = .5f;
      sunshine_depth_statistics::moments moments;
      moments.supplied = moments.valid = true;
      moments.A = direction == raw::orientation::normal ? 1.f : 0.f;
      moments.inverseB = direction == raw::orientation::normal ? -1.f : 1.f;
      moments.count = q.size();
      moments.tiles_x = submitted.grid_width;
      moments.tiles_y = submitted.grid_height;
      for (const double value : q) { moments.sum += value; moments.sum_squares += value * value; }
      raw::selected_frame current;
      current.basis_epoch = submitted.camera.unit_epoch;
      current.layout_epoch = submitted.selection.layout_epoch;
      current.source = submitted.selection.original;
      current.direction = direction;
      current.depth_ready = current.aligned_viewport_assumed = true;
      raw::policy controller;
      require(controller.reset(current.basis_epoch, 1000), "Full-image integration fixture did not reset");
      const double budget = double(scene::reference_zpd) * .5;
      controller.configure({.5, budget, budget});
      binding::mailbox mailbox;
      raw::output state;
      for (unsigned capture = 0; capture != 6; ++capture) {
        const auto now = 1000 + capture * 250;
        submitted.capture_ms = now;
        submitted.camera.frame = {80 + capture, 800};
        current.frame = submitted.camera.frame;
        completion_fixture completed(submitted, start(mailbox, submitted, now));
        for (unsigned y = 0; y != submitted.grid_height; ++y)
          for (unsigned x = 0; x != submitted.grid_width; ++x) {
            const auto px = sunshine_depth_statistics::tile_bounds(x, 192, submitted.grid_width).center();
            const auto py = sunshine_depth_statistics::tile_bounds(y, 108, submitted.grid_height).center();
            const float depth = q[py * 192 + px];
            completed.pixels[y * submitted.grid_width + x] =
              direction == raw::orientation::normal ? 1.f - depth : depth;
          }
        const auto classification = sunshine_depth::analyze_depth(completed.pixels.data(), completed.pixels.size(),
          submitted.grid_width, submitted.grid_height);
        require(classification.kind != sunshine_depth::content_kind::useful,
          "Sparse fixture unexpectedly placed foreground at a selector point");
        binding::associated_sample associated;
        require(mailbox.complete(completed.result, now, associated) == binding::status::associated,
          "Selector classification vetoed authenticated depth transport");
        raw::sample packet;
        packet.id = associated.sample.id;
        packet.capture_ms = associated.sample.capture_ms;
        packet.metadata = current;
        packet.readback_frame = current.frame;
        packet.readback_source = current.source;
        packet.readback_layout_epoch = current.layout_epoch;
        packet.range_supplied = packet.range_valid = true;
        packet.range_min = direction == raw::orientation::normal ? .5f : 0.f;
        packet.range_max = direction == raw::orientation::normal ? 1.f : .5f;
        packet.moments = moments;
        packet.moments.valid = capture != 4;
        state = controller.update(current, &packet, now);
        if (capture == 3 || capture == 5)
          require(state.ready && state.has_depth_statistics && state.depth_statistics.pixel_count == q.size() &&
            state.depth_statistics.mean == moments.sum / moments.count,
            "Authenticated full-image geometry was replaced by empty selector points");
        if (capture == 4)
          require(!state.ready && state.reason == raw::status::invalid_depth && !state.has_depth_statistics,
            "Malformed supplied moments failed to invalidate earlier scene evidence");
      }
    }
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
    dynamic_grid_and_crop_are_frozen();
    sparse_selector_points_do_not_veto_full_image_geometry();
    captured_probe_is_immutable_and_weak();
    probe_capture_rejects_mismatched_envelopes();
    std::puts("PASS camera sample binding: immutable camera/copy/frame, independent request IDs, scope/layout/lifetimes, cancellation, aging and proof boundary");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL camera sample binding: %s\n", e.what());
    return 1;
  }
}
