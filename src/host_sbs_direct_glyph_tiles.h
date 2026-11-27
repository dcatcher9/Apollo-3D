/** @file Tile storage of the native Direct Glyph palette transport. */
#pragma once

namespace models::host_sbs_direct_glyph_gpu {
  struct tile_shape_t {
    unsigned int width, height;

    [[nodiscard]] constexpr unsigned int input_words() const noexcept { return 3u * width * height; }
    [[nodiscard]] constexpr unsigned int output_words() const noexcept { return 4u * width * height; }
  };

  // A 512-pixel-wide tile; the encoder reconstructs the remaining vertical context at quarter resolution.
  inline constexpr tile_shape_t palette_tile {512u, 12u + 28u + 12u};
  inline constexpr unsigned int tile_capacity = 6u;
  inline constexpr unsigned int palette_input_words = tile_capacity * palette_tile.input_words();
  inline constexpr unsigned int palette_output_words = tile_capacity * palette_tile.output_words();
}
