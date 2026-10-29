// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// In the test add-on, the internals of streamline_camera_probe.cpp that its
// runtime-test exports (test_streamline_camera_probe_exports.cpp) reach, and
// the refusal recorders the exports file defines for the probe's refusal
// points. Only those two files include this header.
//
// The shipping add-on keeps them internal, in an anonymous namespace. The test
// add-on (SUNSHINE_SBS_RUNTIME_TEST_ADDON) gives the same definitions external
// linkage in an inline namespace, so that its exports can run in their own
// translation unit of the same DLL; lookup inside the probe file is unchanged.
#include "streamline_camera_probe.h"

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#define SUNSHINE_PROBE_INTERNAL inline namespace probe_test_access
#include "depth_refusal.h"
#include "frame_generation_inputs.h"
#include <cstdint>

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    // Defined in streamline_camera_probe.cpp.
    template<class T> bool read(const T *source, T &destination);

    // Defined in test_streamline_camera_probe_exports.cpp and called where the
    // probe decides a refusal, for SunshineDepthTestLastRefusal
    // (depth_refusal.h). A Streamline tag or evaluation refusal, recorded
    // where its log line is formatted but before the line's 5 s limit.
    void remember_refusal(const evaluation_snapshot &snapshot, sunshine_depth_refusal::gate where, const char *reason,
      const depth_capture::record_diagnostic *capture = nullptr);
    // DLSSG.Depth's refusal; nothing when it copied, the tag copied first or
    // no key was offered.
    void remember_frame_generation_refusal(const frame_generation::attempt &out, std::uint64_t epoch,
      std::uint64_t sequence, std::uint32_t viewport, depth_capture::status status = depth_capture::status::unavailable);
  }
}
#else
#define SUNSHINE_PROBE_INTERNAL namespace
#endif
