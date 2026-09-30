// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"

#include <cstdio>
#include <stdexcept>

namespace {
  namespace api = reshade::api;
  namespace layer = sunshine_game3d::ui_layer;

  void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
  }

  api::resource_desc target(api::format format, std::uint32_t width = 3840, std::uint32_t height = 2160) {
    return api::resource_desc(width, height, 1, 1, format, 1, api::memory_heap::default_, api::resource_usage::render_target);
  }
}

int main() {
  try {
    const float transparent[4]{}, black[4]{0.f, 0.f, 0.f, 1.f}, tinted[4]{0.f, 0.f, 0.001f, 0.f};
    for (const auto format : {api::format::r8g8b8a8_unorm, api::format::r8g8b8a8_typeless, api::format::b8g8r8a8_unorm_srgb,
           api::format::r10g10b10a2_unorm, api::format::r16g16b16a16_float, api::format::r32g32b32a32_float})
      require(layer::qualifies(target(format), transparent, 3840, 2160), "A transparent-cleared output-size alpha target did not qualify");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), black, 3840, 2160), "Opaque black is not a transparent clear");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), tinted, 3840, 2160), "A tinted clear qualified");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm, 2228, 1256), transparent, 3840, 2160),
      "A render-resolution scene target qualified as an output layer");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), transparent, 0, 0), "No output size must reject everything");
    for (const auto format : {api::format::r11g11b10_float, api::format::r8_unorm, api::format::r16g16_float, api::format::b8g8r8x8_unorm})
      require(!layer::qualifies(target(format), transparent, 3840, 2160), "A format without alpha qualified");
    auto msaa = target(api::format::r8g8b8a8_unorm); msaa.texture.samples = 4;
    auto mips = target(api::format::r8g8b8a8_unorm); mips.texture.levels = 2;
    auto array = target(api::format::r8g8b8a8_unorm); array.texture.depth_or_layers = 2;
    auto volume = target(api::format::r8g8b8a8_unorm); volume.type = api::resource_type::texture_3d;
    for (const auto &desc : {msaa, mips, array, volume})
      require(!layer::qualifies(desc, transparent, 3840, 2160), "A multisampled, mipped, layered or 3D target qualified");
    std::puts("PASS UI layer census qualification: output-size single-sample 2D alpha targets cleared to transparent black only");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
