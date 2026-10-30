// SPDX-License-Identifier: GPL-3.0-only
#pragma once

namespace sunshine_game3d::addon_config {
  // Read at the caller's lifecycle point; no cached configuration state.
  bool streamline_depth_source();
  bool upscaler_call_trace();
  bool ngx_depth_source();
  bool queue_fence_order();
  bool frame_activity_trace();
}
