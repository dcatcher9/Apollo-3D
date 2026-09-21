// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "depth_presentation_stats.h"
#include "depth_moments.h"
#include "streamline_depth_capture.h"
#include "depth_cache_update.h"
#include <array>

namespace sunshine_depth { struct sample_result; }

// One ReShade-facing consumer for API-authoritative depth. Native snapshots and
// shared ReShade preservation feed the same binding and calibration readback.
namespace sunshine_streamline::provider {
  // Busy queries retain a confirmed source choice, not stale pixel readiness.
  // Observation loss revokes the old FG scope until the game confirms a mode.
  struct frame_generation_policy {
    depth_capture::selection_policy update(frame_generation_query_status status,
        const frame_generation_snapshot &value) {
      if (status == frame_generation_query_status::observed) {
        selected_ = value.enabled ? depth_capture::selection_policy{true, value.epoch, value.viewport} :
          depth_capture::selection_policy{};
        generated_frames_ = value.enabled ? value.generated_frames : 0;
        confirmed_ = true;
      } else if (status == frame_generation_query_status::busy) {
        return confirmed_ ? selected_ : depth_capture::selection_policy{true};
      } else if (status == frame_generation_query_status::ambiguous || selected_.require_frame_generation) {
        selected_ = {true};
        generated_frames_ = 0;
        confirmed_ = false;
      }
      return selected_;
    }
    std::uint32_t generated_frames() const { return generated_frames_; }
  private:
    depth_capture::selection_policy selected_;
    std::uint32_t generated_frames_{};
    bool confirmed_{};
  };
  struct center_sample {
    sunshine_scene_depth::frame metadata;
    sunshine_depth::frame_depth::projection_t projection;
    std::uint64_t id{}, tick{};
    // Exact point grid for diagnostics only; live scene geometry uses moments.
    std::uint32_t width{}, height{};
    std::array<float, sunshine_depth_statistics::maximum_tiles> raw{};
    bool range_valid{};
    float range_min{}, range_max{};
    sunshine_depth_statistics::moments moments{};
  };
  struct source_description {
    std::uint64_t resource{}, capture{}, sequence{};
    std::uint32_t width{}, height{}, format{}, tag{};
    // Native resource lifetime cookie, shared with ReShade's inventory. The
    // resource address can differ across wrappers or be reused after destruction.
    std::uint64_t identity{};
  };
  struct source_status {
    bool selected{}, ready{};
    bool reused_depth{};
    bool shared_preservation{};
    sunshine_scene_depth::provider_kind provider{sunshine_scene_depth::provider_kind::streamline};
    source_description current, last_valid;
    sunshine_depth_stats::presentation_statistics presentations;
  };
  void initialize(reshade::api::effect_runtime *runtime);
  void destroy(reshade::api::effect_runtime *runtime);
  void reload(reshade::api::effect_runtime *runtime);
  // Native rendering outlives unrelated FX reloads; only refresh FX bindings.
  void reload_effect_bindings(reshade::api::effect_runtime *runtime);
  // Shared by Generic and API depth; reflects once per effects reload and
  // publishes readiness only when it changes or new uniforms need initializing.
  void set_depth_ready(reshade::api::effect_runtime *runtime, bool ready);
  bool complete(reshade::api::effect_runtime *runtime, const sunshine_depth::sample_result &value);
  // Returns true when SL owns this effects pass, even if it must remain mono.
  bool begin(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands,
    std::uint64_t present, bool allowed);
  bool finish(reshade::api::effect_runtime *runtime, reshade::api::command_list *commands);
  bool current(reshade::api::effect_runtime *runtime, sunshine_depth::frame_depth &out);
  bool latest_center(reshade::api::effect_runtime *runtime, center_sample &out);
  bool selected(reshade::api::effect_runtime *runtime);
  // Focus/technique/export invalidation cannot carry approximate held depth
  // across a new viewing interval. The owned texture and numeric history remain.
  void invalidate_reused_depth(reshade::api::effect_runtime *runtime);
  bool uses_shared_preservation(reshade::api::effect_runtime *runtime);
  // Overlay snapshot, valid after effects finish too. Deliberate bounded reuse marks
  // the active previous real capture explicitly; otherwise it is only last_valid.
  source_status describe(reshade::api::effect_runtime *runtime, std::uint64_t present);
}
