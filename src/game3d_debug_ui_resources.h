// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <cstdint>
#include <cstring>

namespace sunshine_game3d::ui_resources {
  enum class provider { streamline, ngx };
  enum class role { color, color_alpha, mask, alpha, motion };
  struct descriptor {
    unsigned artifact_id;
    provider source;
    const char *name;
    const char *file_stem;
    const char *semantic;
    role content;
    unsigned tag_type;
    const char *parameter_key;
  };

  // SDK authority: NVIDIA-RTX/Streamline include/sl_core_types.h (2026-09-19)
  // and NVIDIA/DLSS 374959484e79a640feaba44c93ac8cfb0a03f5b5
  // include/nvsdk_ngx_defs.h. IDs are dump identities, not SDK enum values.
  // A catalog entry proves neither runtime availability nor a usable HUD mask.
  // Transfer functions and rectangles belong to each captured observation.
  inline constexpr descriptor catalog[] {
    {9, provider::streamline, "HUDLessColor", "sl_hudless_color", "color_without_ui", role::color, 2, nullptr},
    {10, provider::streamline, "UIColorAndAlpha", "sl_ui_color_alpha", "ui_color_and_alpha", role::color_alpha, 23, nullptr},
    {11, provider::streamline, "NoWarpMask", "sl_no_warp_mask", "skip_warp_hint", role::mask, 54, nullptr},
    {12, provider::streamline, "InvalidDepthMotionHint", "sl_invalid_depth_motion_hint", "color_depth_motion_mismatch_hint", role::mask, 33, nullptr},
    {13, provider::streamline, "Alpha", "sl_alpha", "generic_alpha_not_ui_mask", role::alpha, 34, nullptr},
    {14, provider::streamline, "ParticleHint", "sl_particle_hint", "particle_hint_not_ui_mask", role::mask, 26, nullptr},
    {15, provider::streamline, "TransparencyHint", "sl_transparency_hint", "transparency_hint_not_ui_mask", role::mask, 27, nullptr},
    {16, provider::streamline, "BiasCurrentColorHint", "sl_bias_current_color_hint", "history_rejection_hint_not_ui_mask", role::mask, 29, nullptr},
    {17, provider::streamline, "ColorBeforeParticles", "sl_color_before_particles", "color_before_particles", role::color, 39, nullptr},
    {18, provider::streamline, "ColorAfterParticles", "sl_color_after_particles", "color_after_particles", role::color, 55, nullptr},
    {19, provider::streamline, "UIAlpha", "sl_ui_alpha", "ui_alpha", role::alpha, 69, nullptr},
    {20, provider::ngx, "TransparencyMask", "ngx_transparency_mask", "transparency_hint_not_ui_mask", role::mask, UINT32_MAX, "TransparencyMask"},
    {21, provider::ngx, "BiasCurrentColorMask", "ngx_bias_current_color_mask", "history_rejection_hint_not_ui_mask", role::mask, UINT32_MAX, "DLSS.Input.Bias.Current.Color.Mask"},
    {22, provider::ngx, "IsParticleMask", "ngx_particle_mask", "particle_hint_not_ui_mask", role::mask, UINT32_MAX, "IsParticleMask"},
    {23, provider::ngx, "AnimatedTextureMask", "ngx_animated_texture_mask", "animated_texture_hint_not_ui_mask", role::mask, UINT32_MAX, "AnimatedTextureMask"},
    {24, provider::ngx, "TransparencyLayer", "ngx_transparency_layer", "transparent_effect_color_not_ui_layer", role::color_alpha, UINT32_MAX, "DLSS.TransparencyLayer"},
    {25, provider::ngx, "TransparencyLayerOpacity", "ngx_transparency_layer_opacity", "transparent_effect_opacity_not_ui_mask", role::alpha, UINT32_MAX, "DLSS.TransparencyLayerOpacity"},
    {26, provider::ngx, "TransparencyLayerMvecs", "ngx_transparency_layer_mvecs", "transparent_effect_motion_not_ui_mask", role::motion, UINT32_MAX, "DLSS.TransparencyLayerMvecs"},
    {27, provider::ngx, "ResponsivityMask", "ngx_responsivity_mask", "responsivity_hint_not_ui_mask", role::mask, UINT32_MAX, "DLSS.ResponsivityMask"},
    {28, provider::ngx, "DisocclusionMask", "ngx_disocclusion_mask", "disocclusion_hint_not_ui_mask", role::mask, UINT32_MAX, "DLSS.DisocclusionMask"},
    {29, provider::streamline, "AnimatedTextureHint", "sl_animated_texture_hint", "animated_texture_hint_not_ui_mask", role::mask, 28, nullptr},
    {30, provider::streamline, "TransparencyLayer", "sl_transparency_layer", "transparent_effect_color_not_ui_layer", role::color_alpha, 51, nullptr},
    {31, provider::streamline, "TransparencyLayerOpacity", "sl_transparency_layer_opacity", "transparent_effect_opacity_not_ui_mask", role::alpha, 52, nullptr},
    {32, provider::streamline, "Backbuffer", "sl_backbuffer", "final_game_color_before_fg_candidate_not_verified_ui_mask", role::color_alpha, 53, nullptr},
  };
  inline constexpr const descriptor *find(unsigned id) noexcept {
    for (const auto &entry : catalog) if (entry.artifact_id == id) return &entry;
    return nullptr;
  }
  inline constexpr const char *artifact_name(unsigned id) noexcept {
    const auto *entry = find(id);
    return entry ? entry->file_stem : nullptr;
  }
  inline constexpr bool is_optional(unsigned id) noexcept { return find(id) != nullptr; }
  inline constexpr const descriptor *find_sl(unsigned type) noexcept {
    for (const auto &entry : catalog) if (entry.source == provider::streamline && entry.tag_type == type) return &entry;
    return nullptr;
  }
  inline const descriptor *find_ngx(const char *name) noexcept {
    if (name) for (const auto &entry : catalog) if (entry.source == provider::ngx && !std::strcmp(entry.parameter_key, name)) return &entry;
    return nullptr;
  }
  inline constexpr const char *role_name(role value) noexcept {
    switch (value) {
      case role::color: return "color";
      case role::color_alpha: return "color_alpha";
      case role::mask: return "mask";
      case role::alpha: return "alpha";
      case role::motion: return "motion";
    }
    return "unknown";
  }
}  // namespace sunshine_game3d::ui_resources
