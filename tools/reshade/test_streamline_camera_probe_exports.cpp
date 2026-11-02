// SPDX-License-Identifier: GPL-3.0-only
// The test add-on's Streamline probe exports, moved from
// streamline_camera_probe.cpp: they drive the real capture boundary with
// actual native resources and command lists.
// The probe's refusal recorders for SunshineDepthTestLastRefusal live here
// too. Built only into SunshineSBSTest; they reach the probe's internals
// through streamline_camera_probe_test_access.h.
#ifndef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#error "Built only into the test add-on (SunshineSBSTest)"
#endif
#include "streamline_camera_probe_test_access.h"
#include "streamline_v1_resource_state.h"
#include "streamline_native_observer.h"
#include "game3d_clock.h"

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
    depth_capture::observe_command(command);
    camera_data copied;
    if (!camera || !read(camera, copied) || !sequence) return 0;
    depth_capture::input value;
    value.epoch = 0x53554e534c544553ull; value.sequence = sequence; value.tick = sunshine_game3d::clock::game_now();
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
  // The queue-order position (Step 4) of the latest positioned
  // ExecuteCommandLists the calling thread made: right after a game
  // submission returns, its copy-out list's if the add-on made one, else its
  // own, as a frame record's depth_read_queue/ticket names a producer.
  extern "C" __declspec(dllexport) BOOL SunshineQueueOrderTestLastExecute(std::uint64_t *queue, std::uint64_t *ticket, BOOL *quiet) {
    if (!queue || !ticket || !quiet) return FALSE;
    const auto value = native_observer::last_execute_position();
    *queue = value.queue; *ticket = value.ticket; *quiet = value.quiet ? TRUE : FALSE;
    return value ? TRUE : FALSE;
  }
}
