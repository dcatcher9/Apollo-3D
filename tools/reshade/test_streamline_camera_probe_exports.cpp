// SPDX-License-Identifier: GPL-3.0-only
// The test add-on's Streamline probe exports, moved from
// streamline_camera_probe.cpp: they drive the real capture boundary, content
// and command association with actual native resources and command lists.
// The probe's refusal recorders for SunshineDepthTestLastRefusal live here
// too. Built only into SunshineSBSTest; they reach the probe's internals
// through streamline_camera_probe_test_access.h.
#ifndef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#error "Built only into the test add-on (SunshineSBSTest)"
#endif
#include "streamline_camera_probe_test_access.h"
#include "streamline_v1_resource_state.h"

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    void remember_refusal(const evaluation_snapshot &snapshot, sunshine_depth_refusal::gate where, const char *reason,
        const depth_capture::record_diagnostic *capture) {
      namespace refusal = sunshine_depth_refusal;
      refusal::record value;
      value.provider = refusal::adapter::streamline; value.where = where;
      value.epoch = snapshot.epoch; value.sequence = snapshot.sequence; value.viewport = snapshot.viewport;
      value.frame_generation = snapshot.frame_generation_input;
      if (capture) {
        value.stage = capture->stage; value.status = capture->result;
        reason = depth_capture::name(capture->stage);
        value.reason = refusal::classify(capture->stage);
      } else value.reason = refusal::classify(reason);
      refusal::remember(value, reason);
    }
    void remember_frame_generation_refusal(const frame_generation::attempt &out, std::uint64_t epoch, std::uint64_t sequence,
        std::uint32_t viewport, depth_capture::status status) {
      namespace refusal = sunshine_depth_refusal;
      const auto where = refusal::gate_of(out.depth);
      if (where == refusal::gate::none) return;
      refusal::record value;
      value.provider = refusal::adapter::dlss_g; value.where = where;
      value.reason = refusal::classify(out.depth, out.depth_stage);
      value.epoch = epoch; value.sequence = sequence; value.viewport = viewport; value.frame_generation = true;
      if (where == refusal::gate::capture) { value.stage = out.depth_stage; value.status = status; }
      refusal::remember(value, where == refusal::gate::capture ? depth_capture::name(out.depth_stage) : frame_generation::name(out.depth));
    }
  }

  static std::uint64_t test_capture(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera,
      std::uint64_t sequence, bool successful, bool version_one, bool defer_finish = false) {
    // Exercise the real capture boundary using actual native resources/CLs.
    // This seam never supplies readiness, selected depth, fences or uniforms.
    test_capture_requested.store(true, std::memory_order_release);
    depth_capture::observe_provider(command);
    camera_data copied;
    if (!camera || !read(camera, copied) || !sequence) return 0;
    depth_capture::input value;
    value.epoch = 0x53554e534c544553ull; value.sequence = sequence; value.tick = GetTickCount64();
    value.observation_revision = depth_observation_revision();
    const auto projection = validate(copied);
    value.projection = {projection.depth_offset, projection.depth_scale,
      projection.valid() && copied.reset == 0, projection.reversed};
    value.native_state = native_state;
    value.proof = version_one && native_state == 0 ? sunshine_scene_depth::state_proof::observed_nonzero :
      sunshine_scene_depth::state_proof::declared;
    value.resource.native = resource;
    value.valid_until = sunshine_scene_depth::lifetime::until_present;
    value.source = depth_capture::retain_source(resource);
    value.source_present_generation = depth_capture::source_present_generation(value.source);
    if (value.source) {
      const auto state = v1_resource_state::resolve(native_state, version_one,
        reinterpret_cast<IUnknown *>(resource));
      value.native_state = state.native;
      value.proof = state.proof;
    }
    depth_capture::begin_evaluation(value.epoch, sequence, 0);
    const auto ticket = depth_capture::nominate(command, value);
    if (ticket && !defer_finish) depth_capture::finish(ticket, successful);
    return ticket;
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestCapture(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool successful) {
    return test_capture(command, resource, native_state, camera, sequence, successful, false);
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestCaptureV1(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool successful) {
    return test_capture(command, resource, native_state, camera, sequence, successful, true);
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestBeginCaptureV1(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool) {
    return test_capture(command, resource, native_state, camera, sequence, false, true, true);
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestFinishCapture(std::uint64_t ticket, bool successful) {
    if (ticket) depth_capture::finish(ticket, successful);
  }
  extern "C" __declspec(dllexport) BOOL SunshinePresentationTestQuery(presentation_snapshot *out) {
    if (!out) return FALSE;
    query_current_presentation(*out);
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineDiscardTestInstallPending() {
    if (!TryAcquireSRWLockExclusive(&poll_lock)) return FALSE;
    const bool result = install_discard_observations();
    ReleaseSRWLockExclusive(&poll_lock);
    return result ? TRUE : FALSE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineDiscardTestCounts(native_discard::counters *out) {
    if (!out) return FALSE;
    *out = native_discard::counts();
    return TRUE;
  }
  extern "C" __declspec(dllexport) void SunshineContentTestEnable(BOOL enabled) {
    test_content_requested.store(enabled != FALSE, std::memory_order_release);
    command_event(command_function([](commands::tracker &) {})); // Serialize epoch transition.
  }
  extern "C" __declspec(dllexport) void SunshineCommandTestLoseLifecycle() {
    command_observation_lost.store(true, std::memory_order_release);
    dropped_observation(loss_diagnostics::reason::test_injected, __func__, __LINE__, {});
  }
  extern "C" __declspec(dllexport) BOOL SunshineCommandTestCapture(std::uint64_t command, commands::recording_marker *out) {
    if (!out) return FALSE;
    *out = capture_command_marker(command);
    return *out ? TRUE : FALSE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineCommandTestAssociate(const commands::recording_marker *evaluation,
      const commands::recording_marker *copy, std::uint64_t queue, commands::association *out) {
    if (!evaluation || !copy || !out) return FALSE;
    *out = associate_commands(*evaluation, *copy, queue);
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineContentTestUse(std::uint64_t command,
      std::uint64_t source, std::uint64_t lifetime, content::use_snapshot *out) {
    if (!out) return FALSE;
    *out = {};
    command_event(command_function([&](commands::tracker &tracker) {
      const auto marker = tracker.mark(command, loss_revision.load(std::memory_order_acquire));
      if (content_observing) *out = content_ledger.use(marker, source, tracker.recording_device(marker));
      if (out->valid() && (!lifetime || out->resource.lifetime != lifetime)) out->state = content::status::lifetime_mismatch;
    }));
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineContentTestAssociate(const content::use_snapshot *use,
      const content::copy_snapshot *copy, std::uint64_t backup, std::uint64_t lifetime, content::association *out) {
    if (!use || !copy || !out) return FALSE;
    *out = associate_content(*use, *copy, use->resource, {backup, lifetime});
    return TRUE;
  }
}
