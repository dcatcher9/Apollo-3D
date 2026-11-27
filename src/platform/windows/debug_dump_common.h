/**
 * @file src/platform/windows/debug_dump_common.h
 * @brief Helpers shared by the host's Dump 3D and Game 3D dump writers.
 *
 * The Game 3D add-on's own dump (tools/reshade) is a separate DLL and does not use this header.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <vector>

// local includes
#include "src/config.h"

namespace platf::debug_dump {

  /** The folder that receives dump packages, and whether APOLLO_SBS_DUMP chose it. */
  struct output_root_t {
    std::filesystem::path path;
    bool from_environment = false;
  };

  /** APOLLO_SBS_DUMP when set, else an "sbs_dump" folder next to the Sunshine log. */
  inline output_root_t output_root() {
    if (const char *value = std::getenv("APOLLO_SBS_DUMP"); value && *value) {
      return {value, true};
    }
    if (!config::sunshine.log_file.empty()) {
      return {std::filesystem::path(config::sunshine.log_file).parent_path() / "sbs_dump", false};
    }
    return {"sbs_dump", false};
  }

  /** The CPU-readable staging description for a full copy of a texture described by `desc`. */
  inline D3D11_TEXTURE2D_DESC readback_staging_desc(D3D11_TEXTURE2D_DESC desc) noexcept {
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    return desc;
  }

  /**
   * Append `row_count` rows of `row_bytes` each, starting at `first_row` of a mapped texture,
   * as tightly packed bytes. A tight row pitch is copied in one piece.
   */
  inline void append_mapped_rows(
    std::vector<std::uint8_t> &bytes,
    const D3D11_MAPPED_SUBRESOURCE &mapped,
    const std::size_t first_row,
    const std::size_t row_count,
    const std::size_t row_bytes
  ) {
    const auto *source = static_cast<const std::uint8_t *>(mapped.pData) + first_row * mapped.RowPitch;
    if (mapped.RowPitch == row_bytes) {
      bytes.insert(bytes.end(), source, source + row_count * row_bytes);
      return;
    }
    for (std::size_t y = 0; y < row_count; ++y) {
      const auto *row = source + y * mapped.RowPitch;
      bytes.insert(bytes.end(), row, row + row_bytes);
    }
  }

}  // namespace platf::debug_dump
