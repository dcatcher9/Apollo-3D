// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file tools/text_hook/text_hook_image.h
 * @brief Whether an image an application loads into a texture is text on nothing.
 *
 * An application with a text layer of its own composes each line in memory and loads the image
 * into a texture. Nothing names that image as text, so it is known by its looks alone. Three
 * measures, each with one meaning, and no knowledge of any engine:
 *
 * - it is mostly clear: text never fills its layer;
 * - what is covered is solid at its core: a glyph is, a veil, a glow or a translucent plate is
 *   not;
 * - what is solid is no larger than a glyph one way or the other, along the row or down the
 *   column. An outline or a shadow joins the strokes of a glyph into one solid shape, and a
 *   heavy one joins the glyphs of a line, so strokes are not asked for: only that the shape
 *   ends within a glyph's size one way. A plate, a picture and a figure are large both ways.
 *
 * Colour is not asked about: text may be of one colour or of many. What is as small as a glyph
 * and is not text passes for it: a thin frame or rule, a narrow bar, a small button or figure.
 * Whoever uses the verdict asks for more than one image before acting on it. The first measure
 * is for an image seen for the first time: an application that keeps its glyphs in one texture
 * sets down more of them as the text goes on, until they fill it. For a texture known to hold
 * text, `glyphs` gives the other two measures alone. Free of any
 * library, so the helper can use it inside an application and the host's tests on images of
 * their own.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace sunshine_text_hook::image {
  /// Text on nothing covers at most this part of its image. A small image may hold one glyph
  /// filling its cell.
  inline constexpr float most_covered = 0.6f, most_covered_small = 0.9f;
  inline constexpr std::uint32_t small_texels = 128u * 128u;
  /// At least this part of what is covered is solid.
  inline constexpr float least_solid = 0.06f;
  /// A glyph with its outline is at most this many texels across, or a twelfth of the image's
  /// shorter side where that is more: a layer made for a larger picture holds larger text.
  inline constexpr std::uint32_t widest_glyph = 64u, glyphs_per_side = 12u;
  /// At least this part of what is solid is that narrow one way or the other.
  inline constexpr float least_narrow = 0.6f;
  /// A texel counts as covered from this coverage on, and as solid from this, or from nine
  /// tenths of the image's fullest coverage where a whole layer is faded.
  inline constexpr unsigned covered_from = 8u, solid_from = 230u;
  /// Fewer covered texels than this among those looked at say nothing either way.
  inline constexpr std::uint32_t least_judged = 32u;
  /// No image has more columns than this, and no more texels than this are looked at.
  inline constexpr std::uint32_t most_columns = 16384u, most_looked_at = 1u << 17u;

  enum class verdict_e : std::uint8_t {
    other,  ///< Not text on nothing: a picture, a sprite, a plate.
    empty,  ///< Nothing at all, as a text layer is between two lines.
    text,
  };

  struct measure_t {
    verdict_e verdict = verdict_e::other;
    /// Tells one image from another by the texels that were looked at.
    std::uint64_t sum = 0u;
    /// True for an image that is plainly a picture: more of it is covered than text ever covers.
    bool dense = false;
    /// True where too little is covered to tell text from anything else: a verdict of `other`
    /// is then no evidence against a texture that held text.
    bool faint = false;
    /// True where what is covered is solid at its core and no larger than a glyph one way,
    /// however much of the image is covered: a store of glyphs that has filled up is still one.
    bool glyphs = false;
    // For a diagnostic log: percent of the texels looked at, of the covered ones, of the solid.
    std::uint32_t covered_percent = 0u, solid_percent = 0u, narrow_percent = 0u;
  };

  /**
   * Look at an image of `width` by `height` texels, `bytes` bytes each with coverage in byte
   * `coverage_at`, at about `budget` of its texels on an even grid. Not for two threads at once.
   */
  inline measure_t examine(const void *data, const std::uint32_t pitch, const std::uint32_t width, const std::uint32_t height, const unsigned bytes, const unsigned coverage_at, const std::uint32_t budget) noexcept {
    measure_t measure;
    if (!data || width == 0u || height == 0u || width > most_columns || bytes == 0u || coverage_at >= bytes || budget == 0u || pitch / bytes < width) {
      return measure;
    }
    const std::uint64_t texels = static_cast<std::uint64_t>(width) * height;
    std::uint32_t step = 1u;
    while (texels / (static_cast<std::uint64_t>(step) * step) > budget || static_cast<std::uint64_t>((width + step - 1u) / step) * ((height + step - 1u) / step) > most_looked_at) {
      ++step;
    }
    // The grid of texels looked at, and a glyph's size on it.
    const std::uint32_t columns = (width + step - 1u) / step, rows = (height + step - 1u) / step;
    const std::uint32_t side = width < height ? width : height;
    const std::uint32_t glyph = side / glyphs_per_side > widest_glyph ? side / glyphs_per_side : widest_glyph;
    const std::uint32_t narrow_run = glyph / step > 0u ? glyph / step : 1u;
    // Per texel looked at: 0 clear, 1 solid, 3 solid and already counted as narrow.
    static std::uint8_t cells[most_looked_at];
    static std::uint16_t down[most_columns];  // per column: the solid texels in a row above
    std::uint32_t seen = 0u, covered = 0u, solid = 0u, narrow = 0u;
    unsigned fullest = 0u, solid_at = solid_from;
    std::uint64_t hash = 0u;
    // Once, or once more where nothing reaches the usual solidity: a layer faded as a whole.
    for (int pass = 0; pass < 2; ++pass) {
      seen = covered = solid = narrow = 0u;
      hash = 14695981039346656037ull;
      for (std::uint32_t column = 0u; column < columns; ++column) {
        down[column] = 0u;
      }
      // A run down a column has ended above `row`: narrow, it counts those of its texels that
      // their run along the row has not counted.
      const auto end_down = [&](const std::uint32_t row, const std::uint32_t column) {
        const std::uint32_t run = down[column];
        down[column] = 0u;
        if (run != 0u && run <= narrow_run) {
          for (std::uint32_t above = 1u; above <= run; ++above) {
            narrow += cells[static_cast<std::size_t>(row - above) * columns + column] != 3u;
          }
        }
      };
      for (std::uint32_t row = 0u; row < rows; ++row) {
        const auto *texels_of_row = static_cast<const std::uint8_t *>(data) + static_cast<std::size_t>(row) * step * pitch;
        std::uint8_t *line = cells + static_cast<std::size_t>(row) * columns;
        std::uint32_t across = 0u;
        const auto end_across = [&](const std::uint32_t column) {
          if (across != 0u && across <= narrow_run) {
            for (std::uint32_t back = 1u; back <= across; ++back) {
              line[column - back] = 3u;
            }
            narrow += across;
          }
          across = 0u;
        };
        for (std::uint32_t column = 0u; column < columns; ++column) {
          const std::uint8_t *texel = texels_of_row + static_cast<std::size_t>(column) * step * bytes;
          const unsigned coverage = texel[coverage_at];
          ++seen;
          if (coverage >= covered_from) {
            ++covered;
            fullest = coverage > fullest ? coverage : fullest;
            // Where and what: the clear texels in between are told by the place of the next.
            hash = (hash ^ seen) * 1099511628211ull;
            for (unsigned byte = 0u; byte < bytes; ++byte) {
              hash = (hash ^ texel[byte]) * 1099511628211ull;
            }
          }
          if (coverage >= solid_at && coverage >= covered_from) {
            ++solid;
            ++across;
            line[column] = 1u;
            down[column] += down[column] != 0xffffu;
          } else {
            line[column] = 0u;
            end_across(column);
            end_down(row, column);
          }
        }
        end_across(columns);
      }
      for (std::uint32_t column = 0u; column < columns; ++column) {
        end_down(rows, column);
      }
      if (solid != 0u || fullest < covered_from || fullest >= solid_from) {
        break;
      }
      solid_at = fullest * 9u / 10u;
    }
    measure.sum = hash;
    measure.covered_percent = covered * 100u / seen;
    measure.solid_percent = covered ? solid * 100u / covered : 0u;
    measure.narrow_percent = solid ? narrow * 100u / solid : 0u;
    if (covered == 0u) {
      measure.verdict = verdict_e::empty;
      return measure;
    }
    measure.faint = covered < least_judged;
    const float limit = texels <= small_texels ? most_covered_small : most_covered;
    measure.dense = static_cast<float>(covered) > limit * static_cast<float>(seen);
    measure.glyphs = !measure.faint && static_cast<float>(solid) >= least_solid * static_cast<float>(covered) &&
                     static_cast<float>(narrow) >= least_narrow * static_cast<float>(solid);
    measure.verdict = measure.glyphs && !measure.dense ? verdict_e::text : verdict_e::other;
    return measure;
  }
}  // namespace sunshine_text_hook::image
