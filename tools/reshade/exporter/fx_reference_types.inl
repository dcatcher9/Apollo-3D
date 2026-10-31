// SPDX-License-Identifier: GPL-3.0-only
// Same TU as exporter.cpp, immediately before runtime_t. The original FX field
// block keeps its member position/order/defaults; effect_name is its string leaf.
// Function-only counterpart: exporter/fx_reference.inl; Generic: generic_fx_reference.inl.
  struct fx_reference_state {
    api::effect_texture_variable texture {};
    std::string effect_name;
    api::effect_technique game_technique {}, native_technique {};
    api::effect_uniform_variable depth_ready {}, calibrated {}, raw_anchor {}, raw_gain {}, depth_rect {}, direction {};
    bool native_uniforms_checked = false;
    bool native_depth_prepared = false;
    api::effect_uniform_variable camera_ready {}, camera_basis {}, camera_projection {}, camera_scale {}, camera_zero {}, camera_blend {}, camera_rect {}, camera_raw_range {};
    api::effect_uniform_variable depth_jitter {}, effect_strength {};
  };

  struct fx_technique_kind {
    bool native, legacy, game;
  };
