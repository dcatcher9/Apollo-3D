// SPDX-License-Identifier: GPL-3.0-only
#pragma once

namespace sunshine_game3d {
  // Full-strength displacement per eye, in source-image horizontal UV units.
  // This is an artistic display budget, not a universal comfort standard.
  // CPU controls and the renderer publish this one value; HLSL consumes b0.
  inline constexpr float default_disparity_limit_uv = 0.01f;
}
