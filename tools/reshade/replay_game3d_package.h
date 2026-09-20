// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "../../src/game3d_debug_formats.h"
#include "../../src/game3d_debug_protocol.h"
#include "../../src/game3d_debug_ui_resources.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dxgiformat.h>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sunshine_game3d::replay {
  inline constexpr std::uint64_t max_package_bytes = game3d_debug::max_capture_bytes;
  inline constexpr unsigned parameter_bytes = 80;

  inline void require(bool condition, const std::string &message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  inline std::uint64_t natural(const nlohmann::json &value, const char *name, std::uint64_t maximum) {
    require(value.is_number_integer() && (value.is_number_unsigned() || value.get<std::int64_t>() >= 0), std::string(name) + " must be a nonnegative integer");
    const auto result = value.get<std::uint64_t>();
    require(result <= maximum, std::string(name) + " exceeds its limit");
    return result;
  }

  inline unsigned bytes_per_pixel(unsigned format) {
    return game3d_debug::pixel_bytes(format);
  }

  inline unsigned artifact_id(const std::string &kind) {
    if (kind == "ui_source_color") return static_cast<unsigned>(game3d_debug::artifact::ui_source_color);
    constexpr std::array<const char *, 8> primary {
      "source_color", "raw_depth", "candidate", "vertical_majorant", "vertical_field", "final_field", "sbs", "linear_color"};
    for (unsigned i = 0; i < primary.size(); ++i)
      if (kind == primary[i]) return i + 1;
    for (const auto &entry : ui_resources::catalog)
      if (kind == entry.file_stem) return entry.artifact_id;
    return 0;
  }

  inline bool color_format(unsigned format) {
    return format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM ||
           format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R16G16B16A16_FLOAT;
  }

  struct artifact {
    std::string kind, file;
    unsigned width = 0, height = 0, format = 0, row_bytes = 0;
    std::uint64_t byte_count = 0;
  };

  struct package {
    unsigned width = 0, height = 0, color = 0;
    bool source_alpha_ui = false;
    std::string ui_alpha_source = "none";
    std::array<std::uint8_t, parameter_bytes> parameters {};
    std::string shader, parameter_abi;
    std::map<std::string, artifact> artifacts;
    // Validated optional captures are diagnostics, not inputs to this renderer.
    std::vector<std::string> ignored_optional_artifacts;
  };

  inline bool needs_disparity_limit_override(const package &p, const std::string &shader) {
    // The same 80-byte layout assigned v1's reserved slot a meaning in v2.
    // Never let a shader experiment interpret the old zero padding as a
    // deliberate zero-strength/mono request. An explicit budget is required.
    return p.parameter_abi == "sunshine_game3d.render_parameters.v1" &&
      shader.find("Sunshine_DisparityLimitUv") != std::string::npos;
  }

  template<class T>
  inline T parameter(const package &p, unsigned offset) {
    static_assert(sizeof(T) == 4);
    T result;
    std::memcpy(&result, p.parameters.data() + offset, sizeof(T));
    return result;
  }

  inline std::string hex(const void *data, std::size_t size) {
    const auto *bytes = static_cast<const std::uint8_t *>(data);
    constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (std::size_t i = 0; i < size; ++i) {
      result[2 * i] = digits[bytes[i] >> 4];
      result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
  }

  inline package parse(const nlohmann::json &manifest) {
    require(manifest.at("schema") == "sunshine.game3d.dump.v1", "Unsupported dump schema");
    require(manifest.at("status") == "complete" || manifest.at("status") == "partial", "Dump does not contain a completed GPU capture");
    const auto &metadata = manifest.at("producer_metadata");
    require(metadata.at("schema") == "sunshine.game3d.gpu-dump.v1", "Unsupported producer schema");
    const auto &record = metadata.at("replay");
    const auto parameter_abi = record.at("parameter_abi").get<std::string>();
    require(record.at("schema") == "sunshine.game3d.replay.v1" &&
      (parameter_abi == "sunshine_game3d.render_parameters.v1" || parameter_abi == "sunshine_game3d.render_parameters.v2") &&
      natural(record.at("parameter_bytes"), "parameter_bytes", parameter_bytes) == parameter_bytes, "Unsupported render parameter ABI");
    require(natural(record.at("compile_flags"), "compile_flags", UINT32_MAX) == 32768 && natural(record.at("compile_flags2"), "compile_flags2", UINT32_MAX) == 0, "Captured compiler flags differ from the production renderer");
    package p;
    p.parameter_abi = parameter_abi;
    // Older packages predate the optional, explicitly selected UI source.
    if (record.contains("source_alpha_ui")) {
      require(record.at("source_alpha_ui").is_boolean(), "source_alpha_ui must be boolean");
      p.source_alpha_ui = record.at("source_alpha_ui").get<bool>();
    }
    p.ui_alpha_source = p.source_alpha_ui ? "source_color" : "none";
    if (record.contains("ui_alpha_source")) {
      require(record.at("ui_alpha_source").is_string(), "ui_alpha_source must be a string");
      p.ui_alpha_source = record.at("ui_alpha_source").get<std::string>();
      require(p.ui_alpha_source == "none" || p.ui_alpha_source == "source_color" || p.ui_alpha_source == "ui_source_color", "Unknown consumed UI-alpha source");
      require(p.source_alpha_ui == (p.ui_alpha_source != "none"), "Consumed UI-alpha source conflicts with effective protection flag");
    }
    p.shader = record.at("shader_source").get<std::string>();
    require(!p.shader.empty() && p.shader.size() <= 256 * 1024 && p.shader.find('\0') == std::string::npos, "Missing or invalid embedded shader");
    const auto &defines = record.at("defines");
    p.width = unsigned(natural(defines.at("BUFFER_WIDTH"), "BUFFER_WIDTH", 8192));
    p.height = unsigned(natural(defines.at("BUFFER_HEIGHT"), "BUFFER_HEIGHT", 8192));
    p.color = unsigned(natural(defines.at("BUFFER_COLOR_SPACE"), "BUFFER_COLOR_SPACE", 3));
    require(p.width && p.height && p.width % 2 == 0 && p.height % 2 == 0 && p.color, "Unsupported render dimensions or color space");
    require(natural(metadata.at("color_space"), "color_space", 3) == p.color, "Conflicting captured color spaces");
    const auto encoded = record.at("parameter_hex").get<std::string>();
    require(encoded.size() == 2 * parameter_bytes, "Incorrect render parameter byte count");
    const auto digit = [](char c) -> unsigned {
      if (c >= '0' && c <= '9') {
        return unsigned(c - '0');
      }
      if (c >= 'a' && c <= 'f') {
        return unsigned(c - 'a' + 10);
      }
      if (c >= 'A' && c <= 'F') {
        return unsigned(c - 'A' + 10);
      }
      throw std::runtime_error("Invalid render parameter hex");
    };
    for (unsigned i = 0; i < parameter_bytes; ++i) {
      p.parameters[i] = std::uint8_t(digit(encoded[2 * i]) * 16 + digit(encoded[2 * i + 1]));
    }
    // Preserve every constant bit, including NaNs and invalid camera domains:
    // reproducing the production shader's guards is part of diagnostic replay.
    const auto &entries = manifest.at("artifacts");
    require(entries.is_array() && entries.size() <= game3d_debug::max_textures, "Invalid artifact list");
    std::uint64_t total = 0;
    std::set<std::string> files, kinds;
    for (const auto &entry : entries) {
      artifact a;
      a.kind = entry.at("kind").get<std::string>();
      a.file = entry.at("file").get<std::string>();
      const auto id = artifact_id(a.kind);
      require(id != 0, "Unknown artifact kind");
      if (entry.contains("artifact_id"))
        require(natural(entry.at("artifact_id"), "artifact_id", UINT32_MAX) == id, "Artifact ID and kind disagree");
      require(kinds.insert(a.kind).second, "Duplicate artifact kind");
      // Host publication uses simple basenames, never directory paths or streams.
      require(!a.file.empty() && a.file != "." && a.file != ".." && a.file.find_first_of("/\\:\"<>|?*") == std::string::npos && a.file.back() != '.' && a.file.back() != ' ', "Artifact path must be a simple filename");
      std::string normalized = a.file;
      for (char &c : normalized) {
        require(static_cast<unsigned char>(c) >= 32, "Artifact filename contains a control character");
        if (c >= 'A' && c <= 'Z') {
          c = char(c - 'A' + 'a');
        }
      }
      const auto stem = normalized.substr(0, normalized.find('.'));
      require(stem != "con" && stem != "prn" && stem != "aux" && stem != "nul" &&
        !(stem.size() == 4 && (stem.compare(0, 3, "com") == 0 || stem.compare(0, 3, "lpt") == 0) && stem[3] >= '1' && stem[3] <= '9'),
        "Artifact filename names a reserved Windows device");
      require(files.insert(normalized).second, "Duplicate artifact file");
      a.width = unsigned(natural(entry.at("width"), "artifact width", game3d_debug::max_dimension));
      a.height = unsigned(natural(entry.at("height"), "artifact height", game3d_debug::max_dimension));
      a.format = unsigned(natural(entry.at("dxgi_format"), "DXGI format", UINT32_MAX));
      a.row_bytes = unsigned(natural(entry.at("row_bytes"), "row_bytes", UINT32_MAX));
      a.byte_count = natural(entry.at("byte_count"), "byte_count", max_package_bytes);
      const auto bpp = bytes_per_pixel(a.format);
      require(a.width && a.height && bpp && a.row_bytes == a.width * bpp && a.byte_count == std::uint64_t(a.row_bytes) * a.height, "Inconsistent artifact dimensions, format or packing");
      require(entry.at("layout") == "tightly packed rows, top to bottom, native little-endian DXGI pixels", "Unsupported artifact packing");
      total += a.byte_count;
      require(total <= max_package_bytes, "Dump exceeds the capture memory limit");
      if (ui_resources::is_optional(id)) {
        // Preserve each optional allocation's own extent/format. Validate it
        // fully even though replay never opens or uploads this diagnostic file.
        p.ignored_optional_artifacts.push_back(a.kind);
        continue;
      }
      if (a.kind == "raw_depth") {
        require(a.format == DXGI_FORMAT_R32_FLOAT, "Raw depth must contain unfiltered float32 samples");
      } else {
        require(a.width == p.width * (a.kind == "sbs" ? 2u : 1u) && a.height == p.height, "Artifact dimensions differ from captured shader defines");
        if (a.kind == "source_color" || a.kind == "ui_source_color") {
          require(color_format(a.format), "Unsupported native source format");
        } else if (a.kind == "sbs") {
          require(a.format == (p.color == 1 ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R16G16B16A16_FLOAT), "Unexpected SBS format");
        } else if (a.kind == "linear_color") {
          require(p.color == 3 && a.format == DXGI_FORMAT_R16G16B16A16_FLOAT, "Unexpected linear color format");
        } else {
          require(a.format == DXGI_FORMAT_R32_FLOAT, "Conditioning fields must contain float32 samples");
        }
      }
      require(p.artifacts.emplace(a.kind, std::move(a)).second, "Duplicate artifact kind");
    }
    require(p.artifacts.count("source_color") && p.artifacts.count("sbs"), "Replay requires captured source color and SBS");
    require(p.artifacts.count("raw_depth") || !parameter<unsigned>(p, 8), "Ready depth has no captured raw texture");
    require(p.artifacts.count("ui_source_color") == (p.ui_alpha_source == "ui_source_color"), "Consumed UI-alpha source requires its exact captured texture; optional SL snapshots cannot substitute");
    return p;
  }
}  // namespace sunshine_game3d::replay
