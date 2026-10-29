// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// TEST add-on only (SunshineGame3DTestLastRender): what the last native
// Game 3D render of a runtime consumed, for runtime fixtures that observe each
// Present with no installed FX. A copy of the depth metadata and render
// constants, never a handle the fixture may sample. The production add-on has
// no such export.
#include "depth_addon.h"
#include "game3d_renderer.h"

#include <array>
#include <cstdint>

namespace sunshine_game3d::test {
  struct last_render {
    // Counts native render attempts that reached depth preparation; rendered
    // is false when that Present's render did not complete.
    std::uint64_t sequence{};
    bool rendered{};
    // The depth this Present's native render consumed: its source identity,
    // allocation, active rectangle and readiness.
    sunshine_depth::frame_depth depth;
    // The render constants the native render used (readiness, scale,
    // convergence and strength blend).
    render_parameters parameters;
    // The frame decision that render placed, for the per-Present trace
    // (test_game3d_present_trace.h): its depth gap (exporter.cpp depth_gap),
    // the real capture its scene was resolved for (frame_decision_t
    // scene_source: epoch, source, sequence, viewport) and the API depth
    // identity of its input (render_frame_input::depth_identity).
    std::uint8_t gap{};
    std::array<std::uint64_t, 4> scene_source{};
    std::uint64_t depth_identity{};
    // The stereo reentry ramp after that Present's scene (runtime_t
    // raw_reentry and reentry_heard): whether the ramp heard from the
    // Present, and its phase and basis (reentry_transition::test_phase,
    // stereo_basis), never its times.
    bool reentry_heard{};
    std::uint8_t reentry_phase{}, reentry_basis{};
  };

  // TEST add-on only (SunshineDepthTestGenericHold): Generic's hold as a
  // runtime's last Present left it (depth_addon.cpp generic_depth_data), for
  // the per-Present trace: whether it held the capture it showed on the
  // Present before, whether that source was selected only to be held, a
  // current copy's readiness, the hold gate's verdict (hold_gate) and the
  // resource identity of the capture shown (0 after a Present that showed no
  // Generic depth).
  struct generic_hold {
    bool held{}, hold_only{}, capture_ready{};
    std::uint8_t verdict{};
    std::uint64_t shown_identity{};
  };
}
