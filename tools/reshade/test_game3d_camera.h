// SPDX-License-Identifier: GPL-3.0-only
// The runtime fixtures' 0/1 environment flags, and the frozen effect's
// preprocessor definitions they select.
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>

namespace sunshine_camera_fixture {
  inline bool flag(const char *name) {
    const char *value = std::getenv(name);
    sunshine_parity::need(!value || !*value || std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
      "Camera fixture flags must be 0 or 1");
    return value && std::strcmp(value, "1") == 0;
  }

  inline bool compiled() { return flag("SUNSHINE_GAME3D_CAMERA_DEPTH") || flag("SUNSHINE_GAME3D_AUTOMATIC"); }

  inline std::string definitions() {
    if (flag("SUNSHINE_GAME3D_AUTOMATIC")) {
      sunshine_parity::need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
        "Automatic setup requires SunshineGame3D");
      return ",SUNSHINE_GAME3D_AUTOMATIC=1";
    }
    const char *value = std::getenv("SUNSHINE_GAME3D_CAMERA_DEPTH");
    if (!value || !*value) return {};
    compiled();
    sunshine_parity::need(std::strcmp(sunshine_depth3d_fixture::effect_file, "SunshineGame3D.fx") == 0,
      "Camera candidate requires SunshineGame3D");
    return std::string(",SUNSHINE_GAME3D_CAMERA_DEPTH=") + value;
  }
}
