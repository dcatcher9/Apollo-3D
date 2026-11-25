// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// One frame's selected depth as its consumers see it: the value type every depth
// source, hold and presenter passes around. It lives apart from depth_addon.h so
// that a source (the API depth provider) need not include the Generic add-on.
#include <Windows.h>
#include <Unknwn.h>
#include <reshade_api.hpp>
#include "depth_orientation.h"
#include "scene_depth_source.h"

#include <cstdint>

namespace sunshine_depth {
  struct frame_depth {
    // Value-only API metadata, separate from the borrowed rendering resources.
    sunshine_scene_depth::frame provided;
    struct projection_t {
      bool supplied = false;
      std::uint64_t epoch = 0;
      std::uint32_t viewport = 0;
      double A = 0, B = 0;
      double raw_scale = 1, raw_bias = 0;
      sunshine_scene_depth::depth_encoding encoding = sunshine_scene_depth::depth_encoding::device;
    } projection;
    // Original game resource, not the add-on's backup copy. Identity-only
    // observation for camera metadata matching; never sampled through this handle.
    reshade::api::resource source_resource {};
    reshade::api::resource resource {};
    reshade::api::resource_view shader_resource {};
    std::uint32_t width = 0, height = 0;
    std::uint32_t x = 0, y = 0, active_width = 0, active_height = 0;
    std::uint64_t source_id = 0, layout_epoch = 0, frame_index = 0;
    std::uint64_t runtime_epoch = 0;
    bool aligned_viewport_assumed = false;
    std::uint64_t backup_id = 0;
    std::uint64_t command_queue = 0;
    bool ready = false;
    // FG may sample the last real frame's private display depth with new color.
    // Its metadata sequence/tick remain original; this is not fresh depth evidence.
    bool reused_depth = false;
    // Provider-resolved FG policy for this source/pass. A busy options query
    // retains confirmed mode; lost or ambiguous scope cannot authorize reuse.
    bool frame_generation_active = false;
    depth_orientation detected_orientation = depth_orientation::automatic;
    depth_orientation orientation = depth_orientation::automatic;
    unsigned orientation_agreeing_frames = 0;
  };
}  // namespace sunshine_depth
