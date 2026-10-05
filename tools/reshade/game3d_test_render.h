// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// TEST add-on only (SunshineGame3DTestLastRender): what the last native
// Game 3D render of a runtime consumed, for runtime fixtures that observe each
// Present with no installed FX. A copy of the depth metadata and render
// constants, never a handle the fixture may sample. The production add-on has
// no such export.
#include "depth_addon.h"
#include "game3d_renderer.h"

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
  };
}
