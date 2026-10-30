// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_depth_input.h"

#include "streamline_depth_provider.h"

namespace sunshine_game3d::depth_input {
  namespace api = reshade::api;

  bool query(api::effect_runtime *runtime, sunshine_depth::frame_depth &out,
      sunshine_depth::depth_orientation orientation, bool request_calibration) {
    return sunshine_depth::get_frame_depth(runtime, out, orientation, request_calibration);
  }

  bool provider_selected(api::effect_runtime *runtime) {
    return sunshine_streamline::provider::selected(runtime);
  }

  bool latest_sample(api::effect_runtime *runtime, sample &out) {
    return sunshine_streamline::provider::latest_center(runtime, out);
  }

  void invalidate_reuse(api::effect_runtime *runtime) {
    sunshine_streamline::provider::invalidate_reused_depth(runtime);
  }

  void set_ready(api::effect_runtime *runtime, bool ready) {
    sunshine_streamline::provider::set_depth_ready(runtime, ready);
  }
}
