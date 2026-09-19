// SPDX-License-Identifier: GPL-3.0-only
// Explicit fixture identity and retired-control contracts for supplied shaders.
#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sunshine_depth3d_fixture {
  inline const char *effect_file = "SuperDepth3D.fx";
  inline const char *technique_name = "SuperDepth3D";

  inline bool expects_sharpening(const char *effect = effect_file) {
    const char *legacy = std::getenv("SUNSHINE_GAME3D_LEGACY_SHARPENING");
    if (legacy && *legacy && std::strcmp(legacy, "0") && std::strcmp(legacy, "1"))
      throw std::runtime_error("SUNSHINE_GAME3D_LEGACY_SHARPENING must be 0 or 1");
    const bool legacy_enabled = legacy && !std::strcmp(legacy, "1");
    const bool game3d = !std::strcmp(effect, "SunshineGame3D.fx");
    if (legacy_enabled && !game3d)
      throw std::runtime_error("Legacy sharpening opt-in is only for frozen SunshineGame3D shaders");
    return !game3d || legacy_enabled;
  }

  template<class Fixture>
  bool sharpening_available(Fixture &fixture, const char *effect = effect_file) {
    const bool expected = expects_sharpening(effect);
    if (bool(fixture.uniform("Sharpen_Power", false).handle) != expected)
      throw std::runtime_error(expected ? "Expected original/legacy sharpening control is missing" :
        "Retired Game3D sharpening control is still reflected");
    return expected;
  }

  // Use the real runtime's preprocessor input, not a rewritten copy of the
  // rendering body. An unset option retains the existing fixture configuration.
  inline std::string configuration_definitions() {
    const char *generic = std::getenv("SUNSHINE_GAME3D_GENERIC_CONFIG");
    if (!generic || !*generic) {
      return {};
    }
    if (std::strcmp(generic, "0") != 0 && std::strcmp(generic, "1") != 0) {
      throw std::runtime_error("SUNSHINE_GAME3D_GENERIC_CONFIG must be 0 or 1");
    }
    if (std::strcmp(effect_file, "SunshineGame3D.fx") != 0) {
      throw std::runtime_error("Generic configuration comparison requires SunshineGame3D");
    }
    return std::string(",SUNSHINE_GAME3D_GENERIC_CONFIG=") + generic;
  }

  inline void select_effect() {
    const char *name = std::getenv("SUNSHINE_DEPTH3D_EFFECT");
    if (!name || !*name || std::strcmp(name, "SuperDepth3D") == 0) {
      effect_file = "SuperDepth3D.fx";
      technique_name = "SuperDepth3D";
    }
    else if (std::strcmp(name, "SunshineGame3D") == 0) {
      effect_file = "SunshineGame3D.fx";
      technique_name = "SunshineGame3D";
    }
    else {
      throw std::runtime_error("SUNSHINE_DEPTH3D_EFFECT must be SuperDepth3D or SunshineGame3D");
    }
  }
}
