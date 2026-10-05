// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "depth_presentation_stats.h"
#include "depth_moments.h"
#include "game3d_depth_input.h"
#include "streamline_depth_capture.h"
#include "depth_cache_update.h"
#include <array>

namespace sunshine_depth { struct sample_result; }

// One ReShade-facing consumer for API-authoritative depth. Native snapshots and
// shared ReShade preservation feed the same binding and calibration readback.
namespace sunshine_streamline::provider {
  // Busy queries retain a confirmed source choice, not stale pixel readiness.
  // Observation loss revokes FG scope while independent ordinary depth remains
  // eligible. This does not turn a failed Off request into confirmed Off.
  struct frame_generation_policy {
    depth_capture::selection_policy update(frame_generation_query_status status,
        const frame_generation_snapshot &value) {
      if (status == frame_generation_query_status::observed) {
        selected_ = value.enabled ? depth_capture::selection_policy{true, value.epoch, value.viewport} :
          depth_capture::selection_policy{};
        generated_frames_ = value.enabled ? value.generated_frames : 0;
        confirmed_ = true;
      } else if (status == frame_generation_query_status::busy) {
        return confirmed_ || selected_.exclude_unconfirmed_fg ? selected_ : depth_capture::selection_policy{true};
      } else if (status == frame_generation_query_status::ambiguous || selected_.require_frame_generation) {
        selected_ = {};
        selected_.exclude_unconfirmed_fg = true;
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
  // The completed sample contract is owned by the neutral presentation input.
  // Keep the adapter's existing internal/test spelling without a second layout.
  using center_sample = sunshine_game3d::depth_input::sample;
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
    sunshine_scene_depth::provider_kind provider{sunshine_scene_depth::provider_kind::streamline};
    source_description current, last_valid;
    sunshine_depth_stats::presentation_statistics presentations;
  };
  // Handoff hysteresis (docs/reshade-sbs.md): when the API source is no
  // longer associated with the queue, API ownership is held in mono instead
  // of starting the generic selector only while all of these hold: the
  // provider owned the previous Present, the last associated Present is at
  // most hold_ms old, the swapchain size is unchanged since then, and some
  // API provider began an evaluation within hold_ms (another source is about
  // to establish). Reload resets the association.
  struct handoff_evidence {
    bool previously_owned{}, evaluation_live{};
    std::uint64_t now_ms{}, associated_ms{};
    std::uint32_t width{}, height{}, associated_width{}, associated_height{};
  };
  constexpr bool handoff_hold(const handoff_evidence &e, std::uint64_t hold_ms) {
    return e.previously_owned && e.associated_ms && e.now_ms >= e.associated_ms && e.now_ms - e.associated_ms <= hold_ms &&
      e.width == e.associated_width && e.height == e.associated_height && e.evaluation_live;
  }
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
  // Overlay snapshot, valid after effects finish too. Deliberate bounded reuse marks
  // the active previous real capture explicitly; otherwise it is only last_valid.
  source_status describe(reshade::api::effect_runtime *runtime, std::uint64_t present);
}
