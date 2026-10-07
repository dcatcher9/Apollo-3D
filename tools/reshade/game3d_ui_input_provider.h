// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_controls.h"
#include "game3d_renderer.h"
#include "game3d_ui_qualification.h"
#include <optional>
#include <string>

namespace sunshine_game3d::ui_input {
  // Only this adapter understands generated-frame input requirements and SDK
  // mask tags. Selection/lifetime remain owned by the bounded capture owner.
  struct presentation {
    frame_generation_mode mode;
    bool observer_active{};
  };
  presentation observe();
  source_alpha_ui_decision resolve(source_alpha_ui_policy &policy, const render_settings &settings,
    const presentation &observed);
  void invalidate(reshade::api::effect_runtime *runtime);
  // Disabling Game 3D invalidates captures but keeps the session's choice.
  // Runtime destruction uses invalidate, which also forgets that choice.
  void suspend(reshade::api::effect_runtime *runtime);
  void configure(reshade::api::effect_runtime *runtime, const source_alpha_ui_decision &status,
    reshade::api::device_api backend);
  void request(reshade::api::effect_runtime *runtime, reshade::api::resource color,
    const source_alpha_ui_decision &status, const render_settings &settings, reshade::api::color_space color_space);
  // Source selection is an optional diagnostic override. Availability is
  // read-only; current-frame GPU checks determine automatic UI protection.
  ui_qualification::status source_status(reshade::api::effect_runtime *runtime);
  void select_source(reshade::api::effect_runtime *runtime, ui_qualification::choice source);

  // One presentation's normalized input. This object owns the observation and
  // diagnostic values; GPU views are borrowed for the current render lease.
  struct frame {
    source_alpha_ui_decision status;
    ui_input_kind kind = ui_input_kind::unavailable;
    reshade::api::resource_view view{};
    ui_mask_channel channel = ui_mask_channel::alpha;
    alpha_auto_source observation;
    // Manual On's explicit first filtered capture (S2): its own provenance,
    // which the adaptive scene match keeps, since the renderer applies that
    // capture directly where detection cannot run. Empty otherwise.
    std::optional<alpha_auto_source> explicit_origin;
    ui_detection_inputs detection;
    bool automatic_detection{};
    std::string source_metadata, capture_metadata;
    // The live UI layer copy this Present offered (ui_layer::live_capture's
    // capture_id; 0 none), reported to the layer's ring by complete().
    std::uint64_t layer_capture{};
    reshade::api::device *layer_device{};
    // A Streamline snapshot was leased on the immediate list this Present
    // (depth_capture::end_local_views ends it after the last read).
    bool leased{};

    ui_adaptive::source match_scene(ui_adaptive::source source, bool scene_ready) const;
    ui_render_input for_render(const ui_plane_parameters &plane, const ui_adaptive::source &adaptive);
    // Claims the renderer's next completion signal for the layer's ring
    // (renderer::claim_completion_value) when this Present offered a copy.
    void complete(renderer &renderer, bool rendered);
  };
  frame acquire(reshade::api::effect_runtime *runtime, renderer &renderer,
    source_alpha_ui_decision status, alpha_auto_policy &session, bool diagnostic);
}
