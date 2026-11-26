// SPDX-License-Identifier: GPL-3.0-only
// Shared helpers of the D3D12 runtime fixtures: the frozen effect's warp
// capability and retired controls, and binary artifact files.
#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace sunshine_parity {
  namespace fs = std::filesystem;
  namespace api = reshade::api;

  inline void need(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  inline bool sole_host_warp(api::effect_runtime *runtime, const char *effect) {
    if (std::strcmp(effect, "SunshineGame3D.fx")) return false;
    bool marked = false;
    runtime->enumerate_uniform_variables(effect, [&](api::effect_runtime *owner, api::effect_uniform_variable uniform) {
      char name[256] {};
      owner->get_uniform_variable_name(uniform, name);
      if (std::strcmp(name, "Depth_Adjustment") && std::strcmp(name, "SunshineGame3D::Depth_Adjustment")) return;
      int version = 0;
      if (owner->get_annotation_int_from_uniform_variable(uniform, "sunshine_warp_version", &version, 1)) {
        need(version == 2, "Unsupported Sunshine warp capability version");
        need(!marked, "Duplicate Sunshine warp capability owner");
        marked = true;
      }
    });
    return marked;
  }

  inline bool retired_ray_control(const char *name) {
    for (const auto retired : {"Compatibility_Power", "Performance_Level", "View_Mode", "View_Mode_Warping",
           "Warping_Masking", "Weapon_Near_Halo_Reduction", "Reconstruction_Size", "De_Artifacting", "Extended_Smoothing"})
      if (!std::strcmp(name, retired)) return true;
    return false;
  }

  inline void write_bytes(const fs::path &path, const void *data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    need(out.good(), "Cannot write parity binary artifact");
  }
}
