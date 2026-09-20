// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "game3d_controls.h"
#include "game3d_renderer.h"

#include <memory>
#include <string>

namespace sunshine_game3d {
  struct diagnostic_frame {
    render_parameters parameters;
    bool source_alpha_ui = false;
    source_alpha_ui_decision source_alpha_decision;
    // Provenance for resources.ui_source, not the latest optional SL snapshot.
    std::string ui_source_metadata;
    automatic_status scene_status;
    sunshine_depth::frame_depth depth;
    diagnostic_resources resources;
    reshade::api::color_space color = reshade::api::color_space::unknown;
    std::uint64_t export_generation = 0, export_sequence = 0, qpc = 0;
    std::uint64_t presentation_ordinal = 0;
    std::uint32_t backbuffer_index = UINT32_MAX;
    std::string_view shader_source;
  };

  // One immutable GPU-only snapshot. Sunshine owns all pixel readback and file
  // publication; the game never blocks on completion or writes image files.
  class debug_dump {
  public:
    debug_dump();
    ~debug_dump();
    bool initialize(std::uint32_t pid, std::uint64_t creation_time);
    // Poll completion and consume cancellations even when rendering is disabled.
    void poll(bool foreground_frame = false);
    // Includes the first arming frame. Avoid scanning foreground windows while
    // no consumer is waiting for a diagnostic capture.
    bool awaiting_capture() const;
    bool requested() const;
    void unavailable(const char *reason);
    void capture(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands, const diagnostic_frame &frame);
    void finish_present(std::uint64_t queue, std::uint64_t swapchain);
    // Called while the runtime/device are still alive, after ReShade's GPU drain.
    void retire_runtime(reshade::api::effect_runtime *runtime);

  private:
    struct impl;
    std::unique_ptr<impl> data_;
  };
}  // namespace sunshine_game3d
