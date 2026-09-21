// SPDX-License-Identifier: GPL-3.0-only
#include "replay_game3d_package.h"

#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>

namespace {
  using namespace sunshine_game3d::replay;
  using sunshine_game3d::ui_parameter_words;
  using sunshine_game3d::ui_plane_mode;
  using sunshine_game3d::ui_plane_parameters;
  using json = nlohmann::json;

  json artifact(const char *kind, unsigned width, unsigned height, unsigned format) {
    return {{"kind", kind}, {"file", std::string(kind) + ".bin"}, {"width", width}, {"height", height}, {"dxgi_format", format}, {"row_bytes", width * bytes_per_pixel(format)}, {"byte_count", std::uint64_t(width) * height * bytes_per_pixel(format)}, {"layout", "tightly packed rows, top to bottom, native little-endian DXGI pixels"}};
  }

  json valid(bool hdr = false) {
    std::array<std::uint8_t, parameter_bytes> parameters {};
    float strength = 50, crop[4] {.1f, .2f, .8f, .7f}, jitter[2] {.0001f, -.0002f};
    unsigned ready = 1;
    std::memcpy(parameters.data(), &strength, 4);
    std::memcpy(parameters.data() + 8, &ready, 4);
    std::memcpy(parameters.data() + 56, jitter, 8);
    std::memcpy(parameters.data() + 64, crop, 16);
    return {{"schema", "sunshine.game3d.dump.v1"}, {"status", "complete"}, {"producer_metadata", {{"schema", "sunshine.game3d.gpu-dump.v1"}, {"color_space", hdr ? 2 : 1}, {"replay", {{"schema", "sunshine.game3d.replay.v1"}, {"parameter_abi", "sunshine_game3d.render_parameters.v1"}, {"parameter_bytes", 80}, {"parameter_hex", hex(parameters.data(), parameters.size())}, {"shader_source", "captured HLSL"}, {"compile_flags", 32768}, {"compile_flags2", 0}, {"defines", {{"BUFFER_WIDTH", 8}, {"BUFFER_HEIGHT", 6}, {"BUFFER_COLOR_SPACE", hdr ? 2 : 1}}}}}}}, {"artifacts", json::array({artifact("source_color", 8, 6, hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM), artifact("raw_depth", 6, 5, DXGI_FORMAT_R32_FLOAT), artifact("candidate", 8, 6, DXGI_FORMAT_R32_FLOAT), artifact("sbs", 16, 6, hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R10G10B10A2_UNORM)})}};
  }

  void reject(const std::function<void(json &)> &change) {
    auto m = valid();
    change(m);
    bool failed = false;
    try {
      (void) parse(m);
    } catch (const std::exception &) {
      failed = true;
    }
    require(failed, "Malformed package was accepted");
  }

  void exact_native_parameters_and_padded_depth() {
    auto m = valid();
    const auto p = parse(m);
    require(p.width == 8 && p.height == 6 && p.color == 1 && p.artifacts.at("raw_depth").height == 5, "Captured render/depth sizes changed");
    require(parameter<float>(p, 56) == .0001f && parameter<float>(p, 60) == -.0002f && parameter<float>(p, 64) == .1f && parameter<float>(p, 76) == .7f, "Jitter/crop constants were changed");
    require(hex(p.parameters.data(), p.parameters.size()) == m["producer_metadata"]["replay"]["parameter_hex"].get<std::string>(), "Parameter byte identity changed");
    auto hdr = parse(valid(true));
    require(hdr.artifacts.at("source_color").format == DXGI_FORMAT_R16G16B16A16_FLOAT, "HDR source converted");
  }

  void malformed_abi_and_shader() {
    reject([](json &m) {
      m["producer_metadata"]["replay"]["parameter_abi"] = "other";
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["parameter_bytes"] = 84;
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["parameter_hex"] = std::string(158, '0');
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["parameter_hex"] = std::string(160, 'x');
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["shader_source"] = "";
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["compile_flags"] = 0;
    });
  }

  void explicit_source_alpha_interpretation() {
    auto m = valid();
    require(!parse(m).source_alpha_ui, "Legacy packages must not implicitly enable source alpha UI");
    for (bool enabled : {true, false}) {
      m["producer_metadata"]["replay"]["source_alpha_ui"] = enabled;
      require(parse(m).source_alpha_ui == enabled, "Captured source alpha boolean changed");
    }
    m["producer_metadata"]["replay"]["source_alpha_ui_requested"] = true;
    m["producer_metadata"]["replay"]["source_alpha_ui_status"] = "unavailable_fg_output_alpha";
    m["producer_metadata"]["replay"]["source_alpha_ui_fg_mode"] = {{"known", true}, {"enabled", true}};
    require(!parse(m).source_alpha_ui, "Replay used the saved preference instead of the consumed FG-safe constant");
    for (const auto &value : json::array({0, 1, -1, .5, "true", "false", nullptr, json::array(), json::object()})) {
      reject([&](json &bad) { bad["producer_metadata"]["replay"]["source_alpha_ui"] = value; });
    }
  }

  void set_ui(json &m, bool enabled, std::uint32_t mode, std::uint32_t inverse_bits) {
    auto &record = m["producer_metadata"]["replay"];
    float inverse;
    std::memcpy(&inverse, &inverse_bits, sizeof(inverse));
    const std::array<std::uint32_t, 4> words {enabled ? 1u : 0u, mode, inverse_bits, 0u};
    record["source_alpha_ui"] = enabled;
    record["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v2";
    record["ui_parameter_bytes"] = ui_parameter_bytes;
    record["ui_parameter_hex"] = hex(words.data(), sizeof(words));
    record["ui_constant_binding"] = {{"register", "b1"}, {"uint32", words}, {"mode", mode},
      {"inverse_depth", std::isfinite(inverse) ? json(inverse) : json(nullptr)}, {"inverse_depth_bits", inverse_bits}};
    record["shader_source"] = "Sunshine_SourceAlphaUI Sunshine_UIPlaneMode";
  }

  void legacy_ui_screen_compatibility() {
    for (const bool enabled : {false, true}) {
      auto m = valid();
      auto &record = m["producer_metadata"]["replay"];
      record["source_alpha_ui"] = enabled;
      const auto legacy = parse(m);
      const auto expected = ui_parameter_words(enabled, {});
      require(legacy.ui_parameter_abi == "legacy_screen" && legacy.ui_plane.mode == ui_plane_mode::screen &&
        legacy.ui_plane.inverse_depth == 0.f &&
        hex(legacy.ui_parameters.data(), legacy.ui_parameters.size()) == hex(expected.data(), sizeof(expected)),
        "Legacy package did not synthesize exact screen-plane b1");
      record["ui_constant_binding"] = {{"register", "b1"}, {"uint32", expected}};
      require(parse(m).ui_parameters == legacy.ui_parameters, "Legacy explicit screen binding changed");
    }
    reject([](json &m) { m["producer_metadata"]["replay"]["ui_parameter_hex"] = std::string(32, '0'); });
    reject([](json &m) { m["producer_metadata"]["replay"]["ui_parameter_bytes"] = 16; });
    reject([](json &m) { m["producer_metadata"]["replay"]["ui_constant_binding"] = {{"register", "b1"}, {"uint32", {0, 1, 0, 0}}}; });
    reject([](json &m) { m["producer_metadata"]["replay"]["ui_constant_binding"] = {{"register", "b1"}, {"uint32", {0, 0, 0, 0}}, {"inverse_depth", 0}}; });
  }

  void exact_independent_ui_plane() {
    // Include signed zero, subnormal/maximum values and malformed production
    // inputs: replay must exercise the shader guards without rewriting bytes.
    for (const auto bits : {0u, 0x80000000u, 1u, 0x3f000001u, 0x7f7fffffu, 0xbf000000u,
           0x7f800000u, 0xff800000u, 0x7fc01234u}) {
      for (const auto mode : {0u, 1u, UINT32_MAX}) {
        for (const bool enabled : {false, true}) {
          auto m = valid();
          const auto original = parse(m).parameters;
          set_ui(m, enabled, mode, bits);
          // JSON serialization must preserve finite float32 roundtrips and use
          // null for nonfinite descriptions while the exact bytes retain them.
          const auto p = parse(json::parse(m.dump()));
          require(p.parameters == original && static_cast<std::uint32_t>(p.ui_plane.mode) == mode &&
            ui_parameter<std::uint32_t>(p, 0) == (enabled ? 1u : 0u) && ui_parameter<std::uint32_t>(p, 8) == bits,
            "Independent UI plane changed captured b0 or b1 bits");
          require(ui_parameter_words(enabled, p.ui_plane)[2] == bits, "Decoded UI plane cannot recreate consumed b1");
          auto changed = original;
          const float value = .125f;
          for (const unsigned offset : {0u, 20u, 52u}) std::memcpy(changed.data() + offset, &value, sizeof(value));
          m["producer_metadata"]["replay"]["parameter_hex"] = hex(changed.data(), changed.size());
          require(parse(m).ui_parameters == p.ui_parameters, "Scene strength/gain/zero changed independent qUI bytes");
        }
      }
    }
  }

  void malformed_independent_ui_metadata() {
    const auto reject_ui = [](const std::function<void(json &)> &change) {
      reject([&](json &m) { set_ui(m, true, 1, 0x3f000000u); change(m["producer_metadata"]["replay"]); });
    };
    reject_ui([](json &r) { r["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v5"; });
    reject_ui([](json &r) { r["ui_parameter_bytes"] = 20; });
    reject_ui([](json &r) { r["ui_parameter_hex"] = std::string(30, '0'); });
    reject_ui([](json &r) { r["ui_parameter_hex"] = std::string(32, 'x'); });
    reject_ui([](json &r) { r["ui_parameter_hex"] = "02000000010000000000003f00000000"; });
    reject_ui([](json &r) { r["source_alpha_ui"] = false; });
    reject_ui([](json &r) { r["ui_parameter_hex"] = "01000000010000000000003f01000000"; });
    reject_ui([](json &r) { r["shader_source"] = "legacy shader"; });
    reject_ui([](json &r) { r.erase("ui_constant_binding"); });
    reject_ui([](json &r) { r["ui_constant_binding"]["register"] = "b0"; });
    reject_ui([](json &r) { r["ui_constant_binding"]["uint32"] = {1, 1, 0}; });
    reject_ui([](json &r) { r["ui_constant_binding"]["uint32"][2] = 0; });
    reject_ui([](json &r) { r["ui_constant_binding"]["mode"] = 0; });
    reject_ui([](json &r) { r["ui_constant_binding"]["mode"] = 1.5; });
    reject_ui([](json &r) { r["ui_constant_binding"]["inverse_depth_bits"] = 0; });
    reject_ui([](json &r) { r["ui_constant_binding"]["inverse_depth"] = .25f; });
    reject_ui([](json &r) { r["ui_constant_binding"]["inverse_depth"] = nullptr; });
    reject_ui([](json &r) { r["ui_constant_binding"]["inverse_depth"] = "0.5"; });
    reject([](json &m) { set_ui(m, true, 1, 0x7f800000u); m["producer_metadata"]["replay"]["ui_constant_binding"]["inverse_depth"] = 1.f; });
  }

  void exact_nearest_ui_plane_contract() {
    for (const bool enabled : {false, true}) {
      auto m = valid();
      set_ui(m, enabled, 2, 0x3f000001u);
      auto &record = m["producer_metadata"]["replay"];
      record["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v3";
      record["shader_source"] = "#define SUNSHINE_UI_NEAREST_PLANE 1\nSunshine_SourceAlphaUI Sunshine_UIPlaneMode";
      const auto p = parse(json::parse(m.dump()));
      require(p.ui_plane.mode == ui_plane_mode::depth_midpoint_nearest_ui &&
          p.ui_parameter_abi == "sunshine_game3d.ui_parameters.v3" &&
          ui_parameter<std::uint32_t>(p, 8) == 0x3f000001u && p.source_alpha_ui == enabled,
        "Nearest UI replay changed the exact submitted floor or protection flag");
    }
    reject([](json &m) { set_ui(m, true, 2, 0x3f000000u); });
    reject([](json &m) {
      set_ui(m, true, 2, 0x3f000000u);
      m["producer_metadata"]["replay"]["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v3";
    });
  }

  void exact_front_limit_ui_plane_contract() {
    for (const bool enabled : {false, true}) {
      for (const auto unused_bits : {0u, 0x3f000001u, 0x7fc12345u}) {
        auto m = valid();
        set_ui(m, enabled, 3, unused_bits);
        auto &record = m["producer_metadata"]["replay"];
        record["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v4";
        record["shader_source"] = "#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1\nSunshine_SourceAlphaUI Sunshine_UIPlaneMode";
        const auto p = parse(json::parse(m.dump()));
        require(p.ui_plane.mode == ui_plane_mode::front_limit && p.source_alpha_ui == enabled &&
            p.ui_parameter_abi == "sunshine_game3d.ui_parameters.v4" &&
            ui_parameter<std::uint32_t>(p, 8) == unused_bits,
          "Front-limit replay changed consumed mode, protection flag, or unused q bytes");
      }
    }
    for (const auto *abi : {"sunshine_game3d.ui_parameters.v2", "sunshine_game3d.ui_parameters.v3"}) {
      reject([&](json &m) {
        set_ui(m, true, 3, 0u);
        auto &record = m["producer_metadata"]["replay"];
        record["ui_parameter_abi"] = abi;
        record["shader_source"] = "#define SUNSHINE_UI_FRONT_LIMIT_PLANE 1\nSunshine_SourceAlphaUI Sunshine_UIPlaneMode";
      });
    }
    reject([](json &m) {
      set_ui(m, true, 3, 0u);
      m["producer_metadata"]["replay"]["ui_parameter_abi"] = "sunshine_game3d.ui_parameters.v4";
    });
  }

  void optional_catalog_stays_out_of_renderer_inputs() {
    auto m = valid(true);
    m["producer_metadata"]["color_space"] = 3;
    m["producer_metadata"]["replay"]["defines"]["BUFFER_COLOR_SPACE"] = 3;
    for (const auto *kind : {"vertical_majorant", "vertical_field", "final_field"})
      m["artifacts"].push_back(artifact(kind, 8, 6, DXGI_FORMAT_R32_FLOAT));
    m["artifacts"].push_back(artifact("linear_color", 8, 6, DXGI_FORMAT_R16G16B16A16_FLOAT));
    const auto baseline = parse(m);
    unsigned index = 0;
    constexpr unsigned formats[] {DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R16G16_SINT,
      DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R11G11B10_FLOAT};
    for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
      auto optional = artifact(entry.file_stem, 3 + index, 2, formats[index % std::size(formats)]);
      optional["artifact_id"] = entry.artifact_id;
      m["artifacts"].push_back(optional);
      ++index;
    }
    require(m["artifacts"].size() > 8 && m["artifacts"].size() <= game3d_debug::max_textures,
      "Fixture must cover the expanded capture capacity");
    const auto p = parse(m);
    require(p.artifacts.size() == 8 && p.ignored_optional_artifacts.size() == index &&
      p.parameters == baseline.parameters && p.shader == baseline.shader && p.color == baseline.color,
      "Optional diagnostics changed replay geometry or entered renderer inputs");
    for (const auto &name : p.ignored_optional_artifacts)
      require(!p.artifacts.count(name), "Optional diagnostics were retained as renderer textures");
    require(p.ignored_optional_artifacts.front() == "sl_hudless_color" &&
      p.ignored_optional_artifacts.back() == "sl_backbuffer", "Ignored optional names were not retained");
    // Kind names also identify captures from manifests without numeric IDs.
    m["artifacts"].back().erase("artifact_id");
    require(parse(m).ignored_optional_artifacts == p.ignored_optional_artifacts, "Optional artifact required redundant numeric ID");
  }

  void exact_external_ui_source_is_required() {
    auto m = valid(true);
    auto &record = m["producer_metadata"]["replay"];
    record["source_alpha_ui"] = true;
    record["ui_alpha_source"] = "ui_source_color";
    auto ui = artifact("ui_source_color", 8, 6, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ui["artifact_id"] = 33;
    m["artifacts"].push_back(ui);
    m["artifacts"].push_back(artifact("sl_backbuffer", 8, 6, DXGI_FORMAT_R10G10B10A2_UNORM));
    auto p = parse(m);
    require(p.ui_alpha_source == "ui_source_color" && p.source_alpha_ui &&
      p.artifacts.count("ui_source_color") && !p.artifacts.count("sl_backbuffer"),
      "Consumed UI source was confused with an independently observed optional backbuffer");
    const auto reject_external = [&](const std::function<void(json &)> &change) {
      auto bad = m;
      change(bad);
      bool failed = false;
      try { (void)parse(bad); } catch (const std::exception &) { failed = true; }
      require(failed, "Invalid or absent consumed UI source was accepted");
    };
    reject_external([](json &bad) { bad["artifacts"].erase(4); });
    reject_external([](json &bad) { bad["artifacts"][4] = artifact("ui_source_color", 4, 6, DXGI_FORMAT_R16G16B16A16_FLOAT); });
    reject_external([](json &bad) { bad["artifacts"][4] = artifact("ui_source_color", 8, 6, DXGI_FORMAT_R32_FLOAT); });
    reject_external([](json &bad) { bad["producer_metadata"]["replay"]["source_alpha_ui"] = false; });
    reject_external([](json &bad) { bad["producer_metadata"]["replay"]["ui_alpha_source"] = "sl_backbuffer"; });
    reject_external([](json &bad) { bad["producer_metadata"]["replay"]["ui_alpha_source"] = "source_color"; });
    reject_external([](json &bad) { bad["producer_metadata"]["replay"]["ui_alpha_source"] = true; });
    reject_external([](json &bad) { bad["artifacts"][4]["artifact_id"] = 32; });
    auto legacy = valid();
    legacy["producer_metadata"]["replay"]["source_alpha_ui"] = true;
    require(parse(legacy).ui_alpha_source == "source_color", "Legacy UI replay did not retain its current color alpha");
    legacy["producer_metadata"]["replay"]["source_alpha_ui"] = false;
    require(parse(legacy).ui_alpha_source == "none", "Legacy disabled UI replay changed");
  }

  void optional_descriptors_are_validated_before_ignoring() {
    const auto reject_optional = [](const std::function<void(json &)> &change) {
      reject([&](json &m) {
        auto optional = artifact("sl_no_warp_mask", 3, 2, DXGI_FORMAT_R8_UINT);
        optional["artifact_id"] = 11;
        change(optional);
        m["artifacts"].push_back(optional);
      });
    };
    for (const auto path : {"../mask.bin", "sub/mask.bin", "sub\\mask.bin", "C:mask.bin", "mask.bin:stream", ".", "..",
           "mask.bin.", "mask.bin ", "NUL.bin", "COM1", "LPT9.bin", "mask?.bin", "SOURCE_COLOR.BIN"}) {
      reject_optional([&](json &a) { a["file"] = path; });
    }
    reject_optional([](json &a) { a["file"] = std::string("mask\0.bin", 9); });
    reject_optional([](json &a) { a["kind"] = "unknown_mask"; });
    reject_optional([](json &a) { a["artifact_id"] = 12; });
    reject_optional([](json &a) { a["artifact_id"] = true; });
    reject_optional([](json &a) { a["width"] = -1; });
    reject_optional([](json &a) { a["width"] = 16385; });
    reject_optional([](json &a) { a["height"] = 0; });
    reject_optional([](json &a) { a["height"] = 2.5; });
    reject_optional([](json &a) { a["row_bytes"] = 4; });
    reject_optional([](json &a) { a["byte_count"] = 5; });
    reject_optional([](json &a) { a["dxgi_format"] = DXGI_FORMAT_R32_TYPELESS; });
    reject_optional([](json &a) { a["layout"] = "bottom-up"; });
    reject([](json &m) {
      m["artifacts"].push_back(artifact("sl_no_warp_mask", 3, 2, DXGI_FORMAT_R8_UINT));
      auto duplicate = m["artifacts"].back();
      duplicate["file"] = "different.bin";
      m["artifacts"].push_back(duplicate);
    });
    reject([](json &m) {
      m["artifacts"].push_back(artifact("sl_hudless_color", 8192, 4096, DXGI_FORMAT_R32G32B32A32_FLOAT));
      m["artifacts"].push_back(artifact("sl_ui_color_alpha", 8192, 4096, DXGI_FORMAT_R32G32B32A32_FLOAT));
    });
    reject([](json &m) {
      while (m["artifacts"].size() <= game3d_debug::max_textures)
        m["artifacts"].push_back(artifact("sl_no_warp_mask", 1, 1, DXGI_FORMAT_R8_UINT));
    });
  }

  void byte_constants_not_reinterpreted() {
    auto m = valid();
    auto p = parse(m);
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(p.parameters.data(), &nan, 4);
    p.parameters[16] = 99;
    m["producer_metadata"]["replay"]["parameter_hex"] = hex(p.parameters.data(), p.parameters.size());
    const auto reread = parse(m);
    require(reread.parameters == p.parameters, "Invalid-domain constants must survive for production guard replay");
  }

  void versioned_disparity_budget() {
    auto manifest = valid();
    const auto legacy = parse(manifest);
    require(parameter<float>(legacy, 28) == 0.f &&
      !needs_disparity_limit_override(legacy, legacy.shader), "Legacy embedded-shader replay must preserve zero padding without requiring a new budget");
    const std::string bounded_shader = "float Sunshine_DisparityLimitUv : packoffset(c1.w);";
    require(needs_disparity_limit_override(legacy, bounded_shader), "New shader cannot silently interpret old reserved bytes as a zero disparity budget");

    auto expected = legacy.parameters;
    const float budget = .01f;
    std::memcpy(expected.data() + 28, &budget, sizeof(budget));
    manifest["producer_metadata"]["replay"]["parameter_abi"] = "sunshine_game3d.render_parameters.v2";
    manifest["producer_metadata"]["replay"]["parameter_hex"] = hex(expected.data(), expected.size());
    manifest["producer_metadata"]["replay"]["shader_source"] = bounded_shader;
    const auto current = parse(manifest);
    require(current.parameters == expected && parameter<float>(current, 28) == budget,
      "V2 budget and the remaining 80-byte parameter block must survive byte-exactly");
    require(!needs_disparity_limit_override(current, current.shader), "V2 replay must consume its recorded budget without an override");

    // Diagnostic replay must preserve bad production-domain bytes too; shader
    // guards, rather than the package loader, determine their rendered output.
    const float invalid_budget = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(expected.data() + 28, &invalid_budget, sizeof(invalid_budget));
    manifest["producer_metadata"]["replay"]["parameter_hex"] = hex(expected.data(), expected.size());
    require(parse(manifest).parameters == expected, "V2 invalid budget bytes must remain reproducible");
    reject([](json &m) { m["producer_metadata"]["replay"]["parameter_abi"] = "sunshine_game3d.render_parameters.v3"; });
  }

  void bounds_and_formats() {
    reject([](json &m) {
      m["artifacts"][0]["width"] = -1;
    });
    reject([](json &m) {
      m["artifacts"][0]["width"] = 8.5;
    });
    reject([](json &m) {
      m["artifacts"][0]["byte_count"] = 191;
    });
    reject([](json &m) {
      m["artifacts"][0]["row_bytes"] = 64;
    });
    reject([](json &m) {
      m["artifacts"][1]["dxgi_format"] = DXGI_FORMAT_R32_TYPELESS;
    });
    reject([](json &m) {
      m["artifacts"][3]["width"] = 8;
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["defines"]["BUFFER_WIDTH"] = 0;
    });
    reject([](json &m) {
      m["producer_metadata"]["replay"]["defines"]["BUFFER_COLOR_SPACE"] = 2;
    });
    reject([](json &m) {
      m["artifacts"][0]["layout"] = "bottom-up";
    });
  }

  void path_and_duplicates() {
    for (const auto path : {"../source.bin", "sub/source.bin", "sub\\source.bin", "C:source.bin", "source.bin:stream", ".", "..", "source.bin.", "source.bin "}) {
      reject([&](json &m) {
        m["artifacts"][0]["file"] = path;
      });
    }
    reject([](json &m) {
      m["artifacts"][0]["file"] = std::string("source\0.bin", 11);
    });
    reject([](json &m) {
      m["artifacts"][1]["file"] = "SOURCE_COLOR.BIN";
    });
    reject([](json &m) {
      m["artifacts"].push_back(m["artifacts"][0]);
    });
  }

  void missing_and_partial() {
    reject([](json &m) {
      m["status"] = "unavailable";
    });
    reject([](json &m) {
      m["artifacts"].erase(1);
    });
    reject([](json &m) {
      m["artifacts"].erase(3);
    });
    auto m = valid();
    m["status"] = "partial";
    m["artifacts"].erase(1);
    auto encoded = m["producer_metadata"]["replay"]["parameter_hex"].get<std::string>();
    encoded.replace(16, 8, "00000000");
    m["producer_metadata"]["replay"]["parameter_hex"] = encoded;
    require(!parse(m).artifacts.count("raw_depth"), "Valid no-depth mono capture rejected");
  }
}  // namespace

int main() {
  try {
    exact_native_parameters_and_padded_depth();
    malformed_abi_and_shader();
    explicit_source_alpha_interpretation();
    legacy_ui_screen_compatibility();
    exact_independent_ui_plane();
    malformed_independent_ui_metadata();
    exact_nearest_ui_plane_contract();
    exact_front_limit_ui_plane_contract();
    exact_external_ui_source_is_required();
    optional_catalog_stays_out_of_renderer_inputs();
    optional_descriptors_are_validated_before_ignoring();
    byte_constants_not_reinterpreted();
    versioned_disparity_budget();
    bounds_and_formats();
    path_and_duplicates();
    missing_and_partial();
    std::puts("PASS: sixteen Game 3D replay package regression groups");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
