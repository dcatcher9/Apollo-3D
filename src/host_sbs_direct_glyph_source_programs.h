/** @file Shader identity of the Direct Glyph source audit. */
#pragma once
#ifdef _WIN32
#include "host_sbs_shader_cache.h"
#include <array>
#include <filesystem>
#include <string_view>

// Shader ownership only. Availability is not GPU completion or draw authority.
namespace models::host_sbs_direct_glyph_source {
namespace cache=host_sbs_shader_cache;
inline constexpr std::array<cache::shader_spec, 4u> direct_source_audit_specs{{
  cache::shader_spec{"direct_source_audit.hlsl","prepare_main","cs_5_0"},
  cache::shader_spec{"direct_source_audit.hlsl","copy_main","cs_5_0"},
  cache::shader_spec{"direct_source_audit.hlsl","finalize_main","cs_5_0"},
  cache::shader_spec{"direct_source_audit.hlsl","envelope_main","cs_5_0"},
}};
inline constexpr std::string_view direct_source_audit_closure="a5145b7de00d113a135b1c852e4a244735d08b9211bafad5bc08f4bc70e47261";
// The audit sources keep their own narrow root below the Host SBS shader directory.
inline std::filesystem::path root(const std::filesystem::path &shader_root){return shader_root/"direct_glyph_source";}
}  // namespace models::host_sbs_direct_glyph_source
#endif
