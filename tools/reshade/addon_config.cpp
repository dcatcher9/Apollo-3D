// SPDX-License-Identifier: GPL-3.0-only
#include "addon_config.h"
#include <Windows.h>
#include <Unknwn.h>
#include <reshade.hpp>

namespace sunshine_game3d::addon_config {
  bool streamline_depth_source() {
    bool enabled = true;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "StreamlineDepthSource", enabled);
    return enabled;
  }

  bool upscaler_call_trace() {
    bool enabled = false;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "UpscalerCallTrace", enabled);
    return enabled;
  }

  bool ngx_depth_source() {
    bool enabled = true;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "NGXDepthSource", enabled);
    return enabled;
  }

  bool queue_fence_order() {
    // Queue order (Step 4; docs/reshade-sbs.md, Queue order): the queue
    // Signal/Wait and fence Signal hooks. 0 turns them off.
    bool enabled = true;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "QueueFenceOrder", enabled);
    return enabled;
  }

  bool frame_activity_trace() {
    bool enabled = false;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "FrameActivityTrace", enabled);
    return enabled;
  }
}
