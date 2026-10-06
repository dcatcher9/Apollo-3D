// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game3d_renderer.h"

namespace sunshine_game3d::test {
  // Keep the established fixture parameter matrices separate from the
  // production input API. Captured input absence never means current alpha.
  // dedicated: an automatic input is a dedicated mask (a declared UI tag)
  // rather than captured colour alpha.
  inline bool render_frame(renderer &target, reshade::api::command_list *commands,
      reshade::api::resource color, reshade::api::resource_view depth,
      const render_parameters &scene, bool enabled = false,
      reshade::api::resource_view mask = {}, const ui_plane_parameters &plane = {},
      const alpha_auto_source *automatic = nullptr, const ui_adaptive::source *adaptive = nullptr,
      ui_mask_channel channel = ui_mask_channel::alpha, bool dedicated = false) {
    render_frame_input frame;
    frame.color = color;
    frame.depth = depth;
    frame.scene = scene;
    frame.ui.view = mask;
    frame.ui.plane = plane;
    frame.ui.automatic = automatic;
    frame.ui.adaptive = adaptive;
    frame.ui.channel = channel;
    if (enabled) {
      if (automatic && dedicated)
        frame.ui.kind = ui_input_kind::dedicated_mask;
      else if (mask.handle || (automatic && automatic->retained))
        frame.ui.kind = ui_input_kind::captured_color_alpha;
      else
        frame.ui.kind = ui_input_kind::current_color_alpha;
    }
    return target.render(commands, frame);
  }
}
