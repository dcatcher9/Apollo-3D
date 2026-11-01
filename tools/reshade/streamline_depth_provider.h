// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "depth_addon.h"
#include "depth_presentation_stats.h"
#include "depth_moments.h"
#include "game3d_depth_input.h"
#include "streamline_depth_capture.h"
#include "depth_hold.h"
#include "freshness.h"
#include <array>

namespace sunshine_depth { struct sample_result; }

// One ReShade-facing consumer for API-authoritative depth. API depth reaches it
// only through native snapshots, which feed the binding and calibration readback.
namespace sunshine_streamline::provider {
  // provider::begin's frame record row (game3d_frame_record.h), in two steps.
  // The acquisition: the FG policy and the scope's selection, the queue's
  // association, the handoff hold and the capture owner's diagnostic. A
  // selected source names the frame selected_metadata describes; otherwise a
  // reported capture names itself.
  sunshine_game3d::api_depth_decision describe_acquisition(const depth_capture::selection_policy &policy,
    const frame_generation_selection &scope, bool associated, bool handoff_hold,
    const depth_capture::capture_diagnostic &capture, const sunshine_scene_depth::frame *selected_metadata);
  // The display that followed: what begin's output holds. Without a selected
  // source or an association there is none (the handoff hold or the release
  // to Generic says why). A fresh copy, else a ready output, which holds an
  // earlier copy that names its own source (held), else the display cache's
  // reason.
  // shown_capture: the capture id of the copy shown, fresh or held.
  void describe_display(sunshine_game3d::api_depth_decision &decision, bool copied, bool output_ready,
    const char *cache_reason, const sunshine_scene_depth::frame &held, std::uint64_t shown_capture = 0);
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
    // The ready depth is an FG scope's own input (not ordinary SR/RR depth).
    bool frame_generation_input{};
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
  // to establish). Reload resets the association. The provider passes
  // freshness.h's handoff_ms.
  struct handoff_evidence {
    bool previously_owned{}, evaluation_live{};
    std::uint64_t now_ms{}, associated_ms{};
    std::uint32_t width{}, height{}, associated_width{}, associated_height{};
  };
  constexpr bool handoff_hold(const handoff_evidence &e, std::uint64_t hold_ms) {
    return e.previously_owned && e.associated_ms && sunshine_freshness::fresh_le(e.associated_ms, e.now_ms, hold_ms) &&
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
