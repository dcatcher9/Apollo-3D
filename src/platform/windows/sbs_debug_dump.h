/**
 * @file src/platform/windows/sbs_debug_dump.h
 * @brief Debug-only: dump the host SBS pipeline's textures (2D source / depth map / SBS result)
 *        to disk for offline inspection of 2D->3D reprojection artifacts. Kept out of
 *        display_vram.cpp so the encode path stays uncluttered.
 */
#pragma once

// platform includes
#include <d3d11.h>

// standard includes
#include <filesystem>

namespace platf::sbs_debug {

  /**
   * @brief Owns the dump destination + a frame counter and performs one-frame texture dumps
   *        when a trigger fires.
   *
   * Triggers: the client "Dump 3D" button (0x3004 control message -> video::sbs_debug_dump_pending)
   * or a "dump.trigger" file appearing in the dump dir (manual fallback). The output dir is
   * APOLLO_SBS_DUMP if set, otherwise an "sbs_dump" folder next to the sunshine log.
   */
  class dumper {
  public:
    dumper();  ///< Resolves the output directory (APOLLO_SBS_DUMP, else <log dir>/sbs_dump).

    bool enabled() const {
      return !dir_.empty();
    }

    /**
     * @brief If a dump is pending, save the current frame's source, depth and SBS-result SRVs
     *        as PNG images (grayscale for R32_FLOAT depth) into a fresh timestamped subfolder.
     *        Any SRV may be null (skipped). Cheap no-op otherwise. Call once per SBS convert().
     */
    void maybe_dump(ID3D11Device *device, ID3D11DeviceContext *ctx,
      ID3D11ShaderResourceView *source,
      ID3D11ShaderResourceView *depth,
      ID3D11ShaderResourceView *sbs,
      bool hdr);

  private:
    std::filesystem::path dir_;
    int counter_ = 0;
    unsigned poll_counter_ = 0;  ///< Rate-limits the dump.trigger file stat to ~1/s.
  };

}  // namespace platf::sbs_debug
