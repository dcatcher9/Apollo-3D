// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_ui_layer.h"
#include "game3d_ui_detection_contract.h"

#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>

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
           api::format::r16g16b16a16_float, api::format::r32g32b32a32_float})
      require(layer::qualifies(target(format), transparent, 3840, 2160), "A transparent-cleared output-size alpha target did not qualify");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), black, 3840, 2160), "Opaque black is not a transparent clear");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), tinted, 3840, 2160), "A tinted clear qualified");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm, 2228, 1256), transparent, 3840, 2160),
      "A render-resolution scene target qualified as an output layer");
    require(!layer::qualifies(target(api::format::r8g8b8a8_unorm), transparent, 0, 0), "No output size must reject everything");
    for (const auto format : {api::format::r11g11b10_float, api::format::r8_unorm, api::format::r16g16_float, api::format::b8g8r8x8_unorm,
           api::format::r10g10b10a2_unorm, api::format::r10g10b10a2_typeless})
      require(!layer::qualifies(target(format), transparent, 3840, 2160), "A format without blended alpha qualified");
    auto msaa = target(api::format::r8g8b8a8_unorm); msaa.texture.samples = 4;
    auto mips = target(api::format::r8g8b8a8_unorm); mips.texture.levels = 2;
    auto array = target(api::format::r8g8b8a8_unorm); array.texture.depth_or_layers = 2;
    auto volume = target(api::format::r8g8b8a8_unorm); volume.type = api::resource_type::texture_3d;
    for (const auto &desc : {msaa, mips, array, volume})
      require(!layer::qualifies(desc, transparent, 3840, 2160), "A multisampled, mipped, layered or 3D target qualified");
    std::puts("PASS UI layer census qualification: output-size single-sample 2D alpha targets cleared to transparent black only");

    // A layer copy in the UI color slot carries the late-layer identity with
    // the premultiplied check; float layers add HDR headroom. Typeless
    // allocations are named by their typeless format.
    namespace detection = sunshine_game3d::ui_detection;
    for (const auto format : {api::format::r8g8b8a8_unorm, api::format::r8g8b8a8_unorm_srgb, api::format::r8g8b8a8_typeless,
           api::format::b8g8r8a8_unorm, api::format::b8g8r8a8_unorm_srgb, api::format::b8g8r8a8_typeless})
      require(layer::detection_flags(format) == 5u &&
          layer::detection_flags(format) == (detection::stored_premultiplied | detection::stored_late_layer),
        "An 8-bit layer lost its late-layer identity or premultiplied check");
    for (const auto format : {api::format::r16g16b16a16_float, api::format::r16g16b16a16_typeless, api::format::r16g16b16a16_unorm,
           api::format::r16g16b16a16_snorm, api::format::r16g16b16a16_uint, api::format::r16g16b16a16_sint,
           api::format::r32g32b32a32_float, api::format::r32g32b32a32_typeless, api::format::r32g32b32a32_uint,
           api::format::r32g32b32a32_sint})
      require(layer::detection_flags(format) == 7u &&
          layer::detection_flags(format) == (detection::stored_premultiplied | detection::stored_hdr_headroom | detection::stored_late_layer),
        "A float layer lost its late-layer identity or HDR headroom");
    // The contract's DXGI test, which ui_detection_replay also uses, names the
    // same float layers as ReShade's typeless families; stored flags never
    // carry a per-frame or reserved bit.
    for (std::uint32_t value = 0; value != 256; ++value) {
      const auto format = static_cast<api::format>(value), typeless = api::format_to_typeless(format);
      if (layer::alpha_format(format))
        require(detection::float_layer_format(value) ==
            (typeless == api::format::r16g16b16a16_typeless || typeless == api::format::r32g32b32a32_typeless),
          "The contract's float layer formats differ from the renderer's typeless families");
      require(!(layer::detection_flags(format) & (detection::per_frame_mask | detection::stored_stage2)),
        "Layer flags set a per-frame or reserved bit");
    }
    std::puts("PASS UI layer detection flags: 5 for 8-bit and 7 for float layers, typeless included; no per-frame bits");

    // game3d_native.hlsl mirrors every flag and sizes detection within the
    // contract's range.
    {
      std::ifstream input(SUNSHINE_GAME3D_NATIVE_HLSL, std::ios::binary);
      require(input.good(), "Cannot read game3d_native.hlsl");
      const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      for (const auto &[name, value] : detection::hlsl_flag_defines) {
        const auto key = "#define " + std::string(name) + ' ';
        const auto at = source.find(key);
        const auto message = std::string(name) + " is missing from game3d_native.hlsl or differs from game3d_ui_detection_contract.h";
        require(at != std::string::npos && std::stoul(source.substr(at + key.size(), 16), nullptr, 0) == value, message.c_str());
      }
      const auto texels = sunshine_game3d::shader_marker(source, detection::decision_texels_marker);
      const auto images = sunshine_game3d::shader_marker(source, detection::scene_evidence_images_marker);
      require(source.find("#define " + std::string(detection::scene_evidence_images_marker) + ' ') != std::string::npos &&
          texels >= detection::min_decision_texels && texels <= detection::max_decision_texels &&
          images <= detection::max_scene_evidence_images,
        "game3d_native.hlsl's UI detection size markers are missing or outside the contract's range");
    }
    std::puts("PASS UI detection contract: game3d_native.hlsl mirrors every flag and sizes detection within range");

    // One game frame: each target cleared in order, then Present.
    const auto frame = [](layer::layer_tracker &tracker, std::initializer_list<std::uint64_t> clears, std::uint64_t now) {
      unsigned captures = 0;
      for (const auto resource : clears) captures += tracker.clear(resource, now) ? 1u : 0u;
      tracker.present(now);
      return captures;
    };
    {
      layer::layer_tracker tracker;
      std::uint64_t now = 1000;
      for (unsigned i = 1; i < layer::confirm_clears; ++i, now += 16)
        require(!frame(tracker, {7}, now) && !tracker.active(), "A layer was trusted before enough clears");
      require(!frame(tracker, {7}, now) && tracker.active() == 7, "A layer cleared every frame was not confirmed");
      now += 16;
      require(frame(tracker, {7, 7, 7}, now) == 1, "The active layer was not copied exactly once per Present");
      layer::layer_tracker burst;
      require(!frame(burst, {8, 8, 8, 8}, now) && !burst.active(), "Clears within one frame confirmed a layer");
      // Of two confirmed layers, the one cleared last in the frame holds the UI.
      for (unsigned i = 0; i < layer::confirm_clears; ++i) frame(tracker, {9, 7}, now += 16);
      require(tracker.active() == 7, "The layer cleared last in the frame was not chosen");
      for (unsigned i = 0; i < layer::confirm_clears; ++i) frame(tracker, {7, 9}, now += 16);
      require(tracker.active() == 9, "The choice did not follow the clear order");
      // A layer that stops being cleared expires.
      tracker.present(now + layer::max_clear_gap_ms + 1);
      require(!tracker.active(), "A layer no longer cleared stayed active");
    }
    {
      layer::layer_tracker tracker;
      std::uint64_t now = 1000;
      // Cleared only now and then: never a per-frame layer.
      for (unsigned i = 0; i < 6; ++i, now += layer::max_clear_gap_ms + 1) frame(tracker, {5}, now);
      require(!tracker.active(), "A target with long gaps between clears was confirmed");
      for (unsigned i = 0; i < layer::confirm_clears; ++i, now += 16) frame(tracker, {1, 2, 3, 4, 5, 6}, now);
      require(tracker.active() == 6, "Several per-frame targets were not all tracked");
      // Many transient targets never displace a confirmed layer.
      std::uint64_t transient = 100;
      for (unsigned i = 0; i < 8; ++i, now += 16)
        frame(tracker, {transient++, transient++, transient++, transient++, transient++, transient++, transient++, transient++, 6}, now);
      require(tracker.active() == 6, "Transient targets displaced the confirmed layer");
      require(!tracker.clear(0, now), "A null resource was tracked");
      tracker.reset();
      require(!tracker.active(), "Reset kept an active layer");
    }
    std::puts("PASS UI layer tracking: confirmed after repeated clears, last-cleared layer active, one copy per Present, expiry");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
