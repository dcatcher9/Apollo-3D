// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_debug_preview.h"
#include "src/game3d_debug_protocol.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <windows.h>

namespace preview = platf::game3d_debug::preview;
using json = nlohmann::json;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: preview_game3d_dump <dump-directory>\n";
    return 2;
  }
  try {
    const auto directory = std::filesystem::canonical(argv[1]);
    const auto manifest_path = directory / "manifest.json";
    if (std::filesystem::file_size(manifest_path) > 1024 * 1024) {
      throw std::runtime_error("Manifest exceeds byte budget");
    }
    std::ifstream stream(manifest_path, std::ios::binary);
    auto manifest = json::parse(stream);
    stream.close();  // Windows replacement requires that our read handle is closed.
    if (manifest.value("schema", "") != "sunshine.game3d.dump.v1") {
      throw std::runtime_error("Not a Game 3D dump");
    }
    const auto &artifacts = manifest.at("artifacts");
    if (!artifacts.is_array() || artifacts.size() > game3d_debug::max_textures) {
      throw std::runtime_error("Invalid artifact list");
    }
    std::vector<std::vector<std::uint8_t>> storage;
    std::vector<preview::image_view> images;
    storage.reserve(artifacts.size());
    std::uint64_t total = 0;
    std::set<std::string> files;
    for (const auto &artifact : artifacts) {
      const auto kind = artifact.at("kind").get<std::string>();
      constexpr const char *primary[] {"source_color", "raw_depth", "candidate", "vertical_majorant",
        "vertical_field", "final_field", "sbs", "linear_color", "ui_source_color"};
      const bool optional = std::none_of(std::begin(primary), std::end(primary),
        [&](const char *name) { return kind == name; });
      try {
        const auto file = artifact.at("file").get<std::string>();
        if (!preview::detail::safe_artifact_filename(file) || !files.insert(file).second) {
          throw std::runtime_error("Unsafe or duplicated artifact filename");
        }
        const auto path = std::filesystem::canonical(directory / file);
        if (path.parent_path() != directory) {
          throw std::runtime_error("Artifact escapes dump directory");
        }
        const auto width = artifact.at("width").get<std::uint32_t>(), height = artifact.at("height").get<std::uint32_t>(), format = artifact.at("dxgi_format").get<std::uint32_t>();
        const auto row = std::uint64_t(width) * preview::detail::bytes_per_pixel(format), size = row * height;
        constexpr std::uint64_t limit = game3d_debug::max_capture_bytes;
        if (!width || !height || width > 16384 || height > 16384 || !row || size > limit || total > limit - size || artifact.at("row_bytes") != row || artifact.at("byte_count") != size || std::filesystem::file_size(path) != size) {
          throw std::runtime_error("Invalid artifact dimensions, packing or byte budget");
        }
        // A failed optional read must not retain unaccounted allocations while
        // later artifacts continue through the same bounded-memory reader.
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        std::ifstream input(path, std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        storage.push_back(std::move(bytes));
        total += size;
        images.push_back({kind, width, height, format, storage.back()});
      } catch (const std::exception &error) {
        if (!optional) throw;
        auto &errors = manifest["optional_capture_errors"];
        if (errors.is_null()) errors = json::array();
        errors.push_back({{"kind", kind}, {"stage", "preview_input"}, {"reason", error.what()}});
      }
    }
    const auto visualizations = preview::generate(directory, manifest, images);
    {
      std::ofstream output(directory / "visualizations.json", std::ios::binary);
      output.exceptions(std::ios::failbit | std::ios::badbit);
      output << visualizations.dump(2) << '\n';
    }
    // Additive only: capture/provenance/replay metadata and native artifacts stay unchanged.
    manifest["visualizations"] = visualizations;
    const auto pending = directory / "manifest.preview.tmp";
    {
      std::ofstream output(pending, std::ios::binary);
      output.exceptions(std::ios::failbit | std::ios::badbit);
      output << manifest.dump(2) << '\n';
    }
    // Windows rename cannot replace an existing file. ReplaceFile is atomic for readers.
    if (!ReplaceFileW(manifest_path.c_str(), pending.c_str(), nullptr, REPLACEFILE_IGNORE_MERGE_ERRORS, nullptr, nullptr)) {
      throw std::runtime_error("Could not publish additive visualization manifest; Win32 error " + std::to_string(GetLastError()));
    }
    std::cout << "Generated " << visualizations.at("images").size() << " PNG previews: " << (directory / "index.html").string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Game 3D previews: " << error.what() << '\n';
    return 2;
  }
}
