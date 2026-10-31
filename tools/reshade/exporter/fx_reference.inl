// SPDX-License-Identifier: GPL-3.0-only
// Function-only part at the end of exporter.cpp; original anonymous namespace.
// FX state: fx_reference_types.inl at its original runtime member slot.
// The same reference path has Generic-side helpers in generic_fx_reference.inl.
namespace {
  bool decode_export_color(const char *output, std::uint32_t input, export_color_t &color) {
    if (input == static_cast<std::uint32_t>(api::color_space::srgb) && std::strcmp(output, "srgb") == 0) {
      color = {api::color_space::srgb, wire::transfer::srgb};
      return true;
    }
    if ((input == static_cast<std::uint32_t>(api::color_space::scrgb) || input == static_cast<std::uint32_t>(api::color_space::hdr10_pq)) && std::strcmp(output, "scrgb") == 0) {
      color = {static_cast<api::color_space>(input), wire::transfer::scrgb};
      return true;
    }
    // 10-bit PQ (Rec.2020, ST 2084) of an HDR10 source; exported only to a
    // consumer that accepts it (frame()).
    if (input == static_cast<std::uint32_t>(api::color_space::hdr10_pq) && std::strcmp(output, "pq") == 0) {
      color = {api::color_space::hdr10_pq, wire::transfer::pq};
      return true;
    }
    return false;
  }

  inline void publisher_t::reset_fx_reference(runtime_t &proof) {
      proof.fx.native_uniforms_checked = false;
      proof.fx.game_technique = proof.fx.native_technique = {};
      proof.fx.depth_ready = proof.fx.calibrated = proof.fx.raw_anchor = proof.fx.raw_gain = proof.fx.depth_rect = proof.fx.direction = {};
      proof.fx.camera_ready = proof.fx.camera_basis = proof.fx.camera_projection = proof.fx.camera_scale = proof.fx.camera_zero = proof.fx.camera_blend = proof.fx.camera_rect = proof.fx.camera_raw_range = proof.fx.depth_jitter = proof.fx.effect_strength = {};
      proof.fx.texture = {}; proof.proof_checked = false;
  }

  inline void publisher_t::refresh_fx_reference(api::effect_runtime *runtime, runtime_t &proof) {
    if (!proof.addon_native && !proof.fx.native_uniforms_checked) {
      proof.fx.native_uniforms_checked = true;
      runtime->enumerate_techniques("SunshineGame3D.fx", [&](api::effect_runtime *owner, api::effect_technique technique) {
        char name[128] {};
        owner->get_technique_name(technique, name);
        if (std::strcmp(name, "SunshineGame3D") == 0) proof.fx.game_technique = technique;
      });
      runtime->enumerate_techniques("SunshineDepth3D.fx", [&](api::effect_runtime *owner, api::effect_technique technique) {
        char name[128] {};
        owner->get_technique_name(technique, name);
        if (std::strcmp(name, "SunshineDepth3D") == 0) proof.fx.native_technique = technique;
      });
      runtime->enumerate_uniform_variables("SunshineDepth3D.fx", [&](api::effect_runtime *owner, api::effect_uniform_variable variable) {
        char qualified[160] {};
        owner->get_uniform_variable_name(variable, qualified);
        const char *name = std::strrchr(qualified, ':');
        name = name ? name + 1 : qualified;
        if (std::strcmp(name, "Sunshine_DepthReady") == 0) proof.fx.depth_ready = variable;
        else if (std::strcmp(name, "Sunshine_Calibrated") == 0) proof.fx.calibrated = variable;
        else if (std::strcmp(name, "Sunshine_RawAnchor") == 0) proof.fx.raw_anchor = variable;
        else if (std::strcmp(name, "Sunshine_RawGain") == 0) proof.fx.raw_gain = variable;
        else if (std::strcmp(name, "Sunshine_DepthRect") == 0) proof.fx.depth_rect = variable;
        else if (std::strcmp(name, "DepthDirection") == 0) proof.fx.direction = variable;
      });
      runtime->enumerate_uniform_variables("SunshineGame3D.fx", [&](api::effect_runtime *owner, api::effect_uniform_variable variable) {
        char qualified[160] {};
        owner->get_uniform_variable_name(variable, qualified);
        const char *name = std::strrchr(qualified, ':');
        name = name ? name + 1 : qualified;
        if (std::strcmp(name, "Sunshine_CameraDepthReady") == 0) proof.fx.camera_ready = variable;
        else if (std::strcmp(name, "Sunshine_CameraCoordinateBasis") == 0) proof.fx.camera_basis = variable;
        else if (std::strcmp(name, "Sunshine_CameraProjection") == 0) proof.fx.camera_projection = variable;
        else if (std::strcmp(name, "Sunshine_CameraDepthScale") == 0) proof.fx.camera_scale = variable;
        else if (std::strcmp(name, "Sunshine_CameraConvergence") == 0) proof.fx.camera_zero = variable;
        else if (std::strcmp(name, "Sunshine_CameraStrengthBlend") == 0) proof.fx.camera_blend = variable;
        else if (std::strcmp(name, "Sunshine_CameraDepthRect") == 0) proof.fx.camera_rect = variable;
        else if (std::strcmp(name, "Sunshine_DepthJitter") == 0) proof.fx.depth_jitter = variable;
        else if (std::strcmp(name, "Sunshine_CameraRawDepthRange") == 0) proof.fx.camera_raw_range = variable;
        else if (std::strcmp(name, "Depth_Adjustment") == 0) proof.fx.effect_strength = variable;
      });
      // Require the source-owned geometry interface; preset mode macros no
      // longer select an alternate renderer.
      proof.raw_supported = proof.fx.camera_ready.handle && proof.fx.camera_basis.handle &&
        proof.fx.camera_projection.handle && proof.fx.camera_scale.handle && proof.fx.camera_zero.handle &&
        proof.fx.camera_blend.handle && proof.fx.camera_rect.handle;

    }
    if (!proof.addon_native && proof.fx.effect_strength.handle)
      runtime->get_uniform_value_float(proof.fx.effect_strength, &proof.frame_strength, 1);
    proof.frame_strength = std::isfinite(proof.frame_strength) ? std::clamp(proof.frame_strength, 0.f, 100.f) : 0.f;
    // Fail closed before reading mutable capture/policy state. Publication
    // below is the only writer that can enable the resolved camera branch.
    if (!proof.addon_native && proof.fx.camera_ready.handle) runtime->set_uniform_value_bool(proof.fx.camera_ready, false);
  }

  inline bool publisher_t::query_fx_reference(api::effect_runtime *runtime, runtime_t &proof, sunshine_depth::frame_depth &depth) {
    const bool game_enabled = proof.addon_native || (proof.fx.game_technique.handle && runtime->get_technique_state(proof.fx.game_technique));
    if (!game_enabled) sunshine_game3d::depth_input::invalidate_reuse(runtime);
    const bool independent_enabled = !game_enabled && proof.fx.native_technique.handle && runtime->get_technique_state(proof.fx.native_technique);
    if (independent_enabled && proof.fx.depth_ready.handle && proof.fx.calibrated.handle && proof.fx.raw_anchor.handle &&
        proof.fx.raw_gain.handle && proof.fx.depth_rect.handle && proof.fx.direction.handle) {
      int direction = 0;
      runtime->get_uniform_value_int(proof.fx.direction, &direction, 1);
      sunshine_game3d::depth_input::query(runtime, depth, direction == 1 ? sunshine_depth::depth_orientation::normal :
        direction == 2 ? sunshine_depth::depth_orientation::reversed : sunshine_depth::depth_orientation::automatic, true);
      apply_native_depth(runtime, proof, depth);
    } else {
      // The original-derived effect owns its preparation and convergence.
      // Camera diagnostics may inspect capture identity without enabling the
      // independent renderer's percentile calibration or altering uniforms.
      const sunshine_game3d::slow_step step("depth input query");
      sunshine_game3d::depth_input::query(runtime, depth);
    }
    return game_enabled;
  }

  inline void publisher_t::publish_fx_reference_jitter(api::effect_runtime *runtime, runtime_t &proof, const frame_decision_t &frame) {
    // Publish from this capture, including on FG reuse. Missing
    // metadata clears the prior offset; it must not borrow the newest jitter.
    if (!proof.addon_native && proof.fx.depth_jitter.handle) runtime->set_uniform_value_float(proof.fx.depth_jitter, frame.jitter.offset.data(), 2);
  }

  inline void publisher_t::publish_fx_reference_scene(api::effect_runtime *runtime, runtime_t &proof, const scene_parameters_t &scene) {
    if (scene.owned && !proof.addon_native) {
      const float strength_ratio = proof.frame_strength > scene.admitted_strength && proof.frame_strength > 0.f ?
        scene.admitted_strength / proof.frame_strength : 1.f;
      if (proof.fx.camera_rect.handle) runtime->set_uniform_value_float(proof.fx.camera_rect, scene.rect.data(), 4);
      if (proof.fx.camera_blend.handle) runtime->set_uniform_value_float(proof.fx.camera_blend, scene.blend * strength_ratio);
      if (scene.ready) {
        runtime->set_uniform_value_int(proof.fx.camera_basis, scene.basis);
        runtime->set_uniform_value_float(proof.fx.camera_projection, scene.projection.data(), 2);
        runtime->set_uniform_value_float(proof.fx.camera_scale, scene.scale);
        runtime->set_uniform_value_float(proof.fx.camera_zero, scene.zero.data(), 2);
        if (proof.fx.camera_raw_range.handle) runtime->set_uniform_value_float(proof.fx.camera_raw_range, scene.raw_range.data(), 2);
        runtime->set_uniform_value_bool(proof.fx.camera_ready, true);
      }
    }
  }

  inline fx_technique_kind publisher_t::classify_fx_reference(api::effect_runtime *runtime, api::effect_technique technique,
      bool addon_render, char (&effect)[128], char (&name)[128]) {
    if (!addon_render) {
      runtime->get_technique_effect_name(technique, effect);
      runtime->get_technique_name(technique, name);
    }
    const bool native = std::strcmp(effect, "SunshineDepth3D.fx") == 0 && std::strcmp(name, "SunshineDepth3D") == 0;
    const bool legacy = std::strcmp(effect, "SuperDepth3D.fx") == 0 && std::strcmp(name, "SuperDepth3D") == 0;
    const bool game = addon_render || (std::strcmp(effect, "SunshineGame3D.fx") == 0 && std::strcmp(name, "SunshineGame3D") == 0);
    return {native, legacy, game};
  }

  inline bool publisher_t::prepare_fx_reference_frame(api::effect_runtime *runtime, runtime_t &proof, const char *effect,
      bool addon_render, bool native, bool legacy, bool game) {
    if (proof.addon_native && !addon_render) return false;
    ring_.release_idle_rings(export_views{*this});
    if (game && proof.raw_supported && !proof.frame.prepared) {
      ring_.deactivate(runtime, "frame_not_prepared");
      return false;
    }
    // The renamed primary effect wins over accidentally enabled reference
    // effects. Never alternate producers between techniques in one present.
    if (!game && proof.fx.game_technique.handle && runtime->get_technique_state(proof.fx.game_technique)) return false;
    if (native && !proof.fx.native_depth_prepared) {
      ring_.deactivate(runtime, "native_depth_unprepared");
      return false;
    }
    // Preserve the older independent-over-reference priority when the renamed
    // primary effect is not enabled.
    if (legacy && proof.fx.native_technique.handle && runtime->get_technique_state(proof.fx.native_technique)) {
      return false;
    }
    if (!addon_render && (!proof.proof_checked || proof.fx.effect_name != effect)) {
      discover(runtime, proof, effect);
    }
    return true;
  }

  inline bool publisher_t::fx_reference_has_rect(const runtime_t &proof) {
    return proof.fx.camera_rect.handle;
  }

  inline bool publisher_t::fx_reference_rejects_linear(const runtime_t &proof, bool linear) {
    return linear && !proof.addon_native;
  }

  inline bool publisher_t::fx_reference_has_raw_range(const runtime_t &proof) {
    return proof.fx.camera_raw_range.handle;
  }

  inline void publisher_t::apply_native_depth(api::effect_runtime *runtime, runtime_t &proof, const sunshine_depth::frame_depth &depth) {
    const bool rectangle = depth.width && depth.height && depth.active_width && depth.active_height &&
      depth.x <= depth.width && depth.y <= depth.height && depth.active_width <= depth.width - depth.x &&
      depth.active_height <= depth.height - depth.y;
    const bool ready = depth.ready && rectangle && depth.shader_resource.handle;
    const bool calibrated = ready && depth.calibrated;
    const float rect[] {
      rectangle ? float(depth.active_width) / depth.width : 1.f,
      rectangle ? float(depth.active_height) / depth.height : 1.f,
      rectangle ? float(depth.x) / depth.width : 0.f,
      rectangle ? float(depth.y) / depth.height : 0.f};
    const float anchor = calibrated ? depth.raw_anchor : 0.f;
    const float gain = calibrated ? depth.raw_gain : 0.f;
    runtime->set_uniform_value_float(proof.fx.raw_anchor, &anchor, 1);
    runtime->set_uniform_value_float(proof.fx.raw_gain, &gain, 1);
    runtime->set_uniform_value_float(proof.fx.depth_rect, rect, 4);
    runtime->set_uniform_value_bool(proof.fx.calibrated, calibrated);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    depth_events::readiness(runtime, depth_events::site_id::native_ready, ready);
#endif
    sunshine_game3d::depth_input::set_ready(runtime, ready);
    proof.fx.native_depth_prepared = true;
  }

  inline void publisher_t::discover(api::effect_runtime *runtime, runtime_t &proof, const char *effect) {
    proof.fx.texture = {};
    proof.fx.effect_name = effect;
    proof.proof_checked = true;
    unsigned exports = 0;
    runtime->enumerate_texture_variables(effect, [&](api::effect_runtime *owner, api::effect_texture_variable texture) {
      int enabled = 0;
      unsigned width = 0, height = 0, input_color = 0;
      char layout[32] {}, color[32] {};
      export_color_t export_color;
      if (!owner->get_annotation_int_from_texture_variable(texture, "sunshine_sbs_export", &enabled, 1) || enabled != 1 || !owner->get_annotation_string_from_texture_variable(texture, "sunshine_sbs_layout", layout) || std::strcmp(layout, "sbs_lr") != 0 || !owner->get_annotation_string_from_texture_variable(texture, "sunshine_sbs_color_space", color) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_color_space", &input_color, 1) || !decode_export_color(color, input_color, export_color) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_width", &width, 1) || !owner->get_annotation_uint_from_texture_variable(texture, "sunshine_sbs_source_height", &height, 1)) {
        return;
      }
      if (!width || width > wire::max_source_width || width % 2 || !height || height > wire::max_source_height || height % 2) {
        return;
      }
      proof.fx.texture = texture;
      ++exports;
      proof.width = width;
      proof.height = height;
      proof.color = export_color;
    });
    if (exports != 1) proof.fx.texture = {};
    if (!proof.fx.texture.handle) {
      log(reshade::log::level::warning, "Sunshine SBS: effect needs exactly one compatible stereo export texture; reinstall the matching Sunshine shader");
    }
  }

  inline bool publisher_t::describe(api::effect_runtime *runtime, runtime_t &proof, source_t &source) {
    std::uint32_t width = 0, height = 0;
    runtime->get_screenshot_width_and_height(&width, &height);
    if (width != proof.width || height != proof.height) {
      return false;
    }
    if (proof.addon_native) source.resource = proof.native_output;
    else {
      api::resource_view view {};
      runtime->get_texture_binding(proof.fx.texture, &view, nullptr);
      if (!view.handle) return false;
      source.resource = runtime->get_device()->get_resource_from_view(view);
    }
    if (!source.resource.handle) {
      return false;
    }
    source.width = width;
    source.height = height;
    source.color = proof.color;
    source.window = static_cast<HWND>(runtime->get_hwnd());
    // ReShade's aggregate-return getters are not MSVC/MinGW ABI compatible. The build
    // adapts only get_resource_from_view; native COM supplies the other descriptions,
    // including MinGW's explicit aggregate-return wrapper for D3D12 GetDesc.
    if (runtime->get_device()->get_api() == api::device_api::d3d11) {
      D3D11_TEXTURE2D_DESC desc {};
      reinterpret_cast<ID3D11Texture2D *>(source.resource.handle)->GetDesc(&desc);
      if (desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1) {
        return false;
      }
      source.format = typed_format(desc.Format);
      if (!proof.adapter_known) {
        com_ptr<IDXGIDevice> device;
        com_ptr<IDXGIAdapter> adapter;
        DXGI_ADAPTER_DESC adapter_desc {};
        auto *native = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
        if (FAILED(native->QueryInterface(IID_PPV_ARGS(device.put()))) || FAILED(device->GetAdapter(adapter.put())) || FAILED(adapter->GetDesc(&adapter_desc))) {
          return false;
        }
        std::memcpy(&proof.adapter_luid, &adapter_desc.AdapterLuid, sizeof(proof.adapter_luid));
        proof.adapter_known = true;
      }
    } else {
      const auto desc = reinterpret_cast<ID3D12Resource *>(source.resource.handle)->GetDesc();
      if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width * 2 || desc.Height != height || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) {
        return false;
      }
      source.format = typed_format(desc.Format);
      if (!proof.adapter_known) {
        auto *native = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native());
        const LUID luid = native->GetAdapterLuid();
        std::memcpy(&proof.adapter_luid, &luid, sizeof(proof.adapter_luid));
        proof.adapter_known = true;
      }
    }
    source.adapter = proof.adapter_luid;
    return wire::supported_format(static_cast<std::uint32_t>(source.format), source.color.output);
  }

} // namespace
