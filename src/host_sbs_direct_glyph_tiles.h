/** @file Tile storage shared by the direct glyph transport and its CUDA graph. */
#pragma once

namespace models::host_sbs_direct_glyph_gpu {
  enum class tile_profile_e : unsigned int { common, small };
  inline constexpr unsigned int tile_profile_count = 2u;

  struct tile_shape_t {
    unsigned int width, height;

    [[nodiscard]] constexpr unsigned int input_words() const noexcept { return 3u * width * height; }
    [[nodiscard]] constexpr unsigned int output_words() const noexcept { return 4u * width * height; }

    [[nodiscard]] constexpr bool fits_band(unsigned int band_width, unsigned int band_height,
                                           unsigned int crop_top) const noexcept {
      const auto aligned_width = (band_width + 3u) & ~3u;
      const auto aligned_height = (band_height + (crop_top & 3u) + 3u) & ~3u;
      return aligned_width >= width && aligned_height >= height;
    }
  };

  [[nodiscard]] constexpr tile_shape_t tile_shape(tile_profile_e profile) noexcept {
    return profile == tile_profile_e::small ? tile_shape_t {384u, 104u} : tile_shape_t {512u, 144u};
  }

  [[nodiscard]] constexpr tile_profile_e tile_profile_for_band(unsigned int width, unsigned int band_height,
                                                               unsigned int crop_top) noexcept {
    if (!tile_shape(tile_profile_e::common).fits_band(width, band_height, crop_top) &&
         tile_shape(tile_profile_e::small).fits_band(width, band_height, crop_top)) {
      return tile_profile_e::small;
    }
    return tile_profile_e::common;
  }

  inline constexpr auto common_tile = tile_shape(tile_profile_e::common);
  inline constexpr auto small_tile = tile_shape(tile_profile_e::small);
  static_assert(small_tile.input_words() <= common_tile.input_words() &&
                small_tile.output_words() <= common_tile.output_words());
  static_assert(tile_profile_for_band(640u, 107u, 253u) == tile_profile_e::small);
  static_assert(tile_profile_for_band(896u, 150u, 354u) == tile_profile_e::common);
  inline constexpr unsigned int tile_width = common_tile.width;
  inline constexpr unsigned int tile_height = common_tile.height;
  inline constexpr unsigned int tile_capacity = 6u;
  inline constexpr unsigned int tile_input_words = common_tile.input_words();
  inline constexpr unsigned int tile_output_words = common_tile.output_words();
  inline constexpr unsigned int placement_words = 4u + 8u * tile_capacity;
  inline constexpr unsigned int placement_scale_word = 10u;
  inline constexpr unsigned int input_atlas_words = tile_capacity * tile_input_words;
  // Slice zero is the fixed TensorRT binding and holds the last tile. Other
  // tiles are saved in the next slice immediately before scoring overwrites it.
  inline constexpr unsigned int output_atlas_words = tile_capacity * tile_output_words;
}
