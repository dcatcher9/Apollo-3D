/** @file Build the resources for Windows display and production Host SBS shaders. */

#include "src/host_sbs_bytecode_groups.h"
#include "src/host_sbs_direct_glyph_publication_gpu.h"
#include "src/host_sbs_shared_ui_warp_runtime.h"
#include "src/logging.h"
#include "src/platform/windows/display_shader_manifest.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <system_error>
#include <windows.h>

namespace {
  namespace fs = std::filesystem;
  namespace cache = models::host_sbs_shader_cache;
  namespace groups = models::host_sbs_bytecode_groups;
  namespace shared_ui = models::host_sbs_shared_ui_warp;
  namespace glyph_publication = models::host_sbs_direct_glyph_publication;

  void finish_output(std::ofstream &output, const fs::path &temporary, const fs::path &destination) {
    output.close();
    if (!output) {
      throw std::runtime_error("Could not write " + temporary.string());
    }
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING)) {
      throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "Could not publish shader build output");
    }
  }

  void write_depfile_path(std::ostream &output, const fs::path &path) {
    for (const char8_t character : path.generic_u8string()) {
      if (character == u8' ' || character == u8'#') {
        output.put('\\');
      } else if (character == u8'$') {
        output.put('$');
      }
      output.put(static_cast<char>(character));
    }
  }

  void write_depfile(const fs::path &path, const fs::path &resource, const std::set<fs::path> &dependencies) {
    auto temporary = path;
    temporary += L".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    write_depfile_path(output, resource);
    output << ':';
    for (const auto &source : dependencies) {
      output << " \\\n  ";
      write_depfile_path(output, source);
    }
    output << '\n';
    finish_output(output, temporary, path);
  }

  void write_shader(std::ostream &output, const std::string &resource_name, std::span<const unsigned char> code) {
    // DXBC chunks and Shader Model 5 instructions are DWORD-aligned. Inline resource words
    // keep the RC script self-contained and avoid one generated side file per shader.
    if (code.empty() || code.size() % 4u != 0u) {
      throw std::runtime_error("Unexpected DXBC size for " + resource_name);
    }
    output << '"' << resource_name << "\" RCDATA\nBEGIN";
    for (std::size_t offset = 0u; offset < code.size(); offset += 4u) {
      if (offset != 0u) {
        output << ',';
      }
      output << (offset % 32u == 0u ? "\n  " : " ");
      const auto word = static_cast<std::uint32_t>(code[offset]) |
                        (static_cast<std::uint32_t>(code[offset + 1u]) << 8u) |
                        (static_cast<std::uint32_t>(code[offset + 2u]) << 16u) |
                        (static_cast<std::uint32_t>(code[offset + 3u]) << 24u);
      output << "0x" << word << 'L';
    }
    output << "\nEND\n\n";
  }

  void embed_shaders(const fs::path &shader_root, const fs::path &resource, const fs::path &depfile) {
    std::vector<groups::bytecode_request_t> bytecodes;
    for (const auto &set : cache::production_prewarm_sets) {
      groups::authenticate_group(set.name, shader_root, set.specs, set.closure, bytecodes);
    }
    // P010 remains independent of startup readiness. Its existing runtime loader resolves this
    // distinct resource only when creating the optional HDR luma program.
    groups::authenticate_group(
      "P010 luma",
      shader_root,
      cache::parallax_v2_p010_y_specs,
      cache::parallax_v2_p010_y_source_closure_sha256,
      bytecodes
    );
    groups::authenticate_group(
      "completed shared UI",
      shader_root,
      shared_ui::completed_specs,
      shared_ui::completed_closure,
      bytecodes
    );
    groups::authenticate_group(
      "subtitle empty gate", shader_root, glyph_publication::empty_gate_shader_specs,
      glyph_publication::empty_gate_shader_source_closure_sha256, bytecodes
    );
    groups::authenticate_group(
      "direct subtitle publication", shader_root, cache::direct_glyph_plan_specs,
      cache::direct_glyph_plan_source_closure_sha256, bytecodes
    );
    groups::authenticate_group(
      "direct subtitle score transport", shader_root, cache::direct_glyph_transport_specs,
      cache::direct_glyph_transport_source_closure_sha256, bytecodes
    );
    groups::authenticate_group(
      "native subtitle palette", shader_root, cache::direct_glyph_palette_specs,
      cache::direct_glyph_palette_source_closure_sha256, bytecodes
    );
    groups::authenticate_group(
      "direct subtitle draw", shader_root, cache::direct_glyph_prepare_specs,
      cache::direct_glyph_prepare_source_closure_sha256, bytecodes
    );
    groups::authenticate_group(
      "Game 3D text UI",
      shader_root,
      shared_ui::text_game_specs,
      shared_ui::text_game_closure,
      bytecodes
    );
    // Display shaders use the build's source closure as their resource key. Runtime can
    // still compile an edited display shader without changing any pinned Host SBS contract.
    const auto display_sources = cache::snapshot_sources(shader_root, platf::dxgi::display_shader_specs);
    if (!display_sources) {
      throw std::runtime_error("Could not snapshot Windows display shader sources");
    }
    for (const auto &spec : platf::dxgi::display_shader_specs) {
      bytecodes.push_back({display_sources, spec, {}, {}});
    }
    groups::prepare_bytecodes(bytecodes);

    auto temporary = resource;
    temporary += L".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Could not open " + temporary.string());
    }
    output << std::hex;
    std::set<std::string> names;
    std::set<fs::path> dependencies;
    for (const auto &request : bytecodes) {
      const auto name = cache::bytecode_resource_name(request.sources, request.spec);
      if (name.size() != 64u || !names.insert(name).second) {
        throw std::runtime_error("Invalid or duplicate shader resource name");
      }
      const auto &code = groups::prepared_bytecode(request, request.spec.filename);
      write_shader(output, name, *code);
      const auto paths = cache::source_paths(request.sources);
      dependencies.insert(paths.begin(), paths.end());
    }
    // Publish the dependency list before the RC target so a failed depfile write leaves the
    // output as it was before the build attempt.
    write_depfile(depfile, resource, dependencies);
    finish_output(output, temporary, resource);
    std::cout << "Prepared " << names.size() << " Windows display and Host SBS shader resources\n";
  }
}  // namespace

int wmain(int argc, wchar_t **argv) {
  if (argc != 4) {
    std::cerr << "Usage: embed-host-sbs-shaders <shader-root> <output.rc> <output.d>\n";
    return 2;
  }
  try {
    auto log = logging::init(3, {});
    embed_shaders(argv[1], argv[2], argv[3]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Could not embed Windows display and Host SBS shaders: " << error.what() << '\n';
    return 1;
  }
}
