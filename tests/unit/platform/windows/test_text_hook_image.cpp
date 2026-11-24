// SPDX-License-Identifier: GPL-3.0-only
#include "tools/text_hook/text_hook_image.h"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {
  namespace image = sunshine_text_hook::image;

  /** A BGRA image a text layer could hold, built from the shapes such layers are made of. */
  struct canvas_t {
    std::uint32_t width, height;
    std::vector<std::uint8_t> pixels;

    canvas_t(const std::uint32_t width, const std::uint32_t height):
        width(width),
        height(height),
        pixels(static_cast<std::size_t>(width) * height * 4u) {}

    void fill(const int left, const int top, const int right, const int bottom, const std::uint8_t coverage, const std::uint8_t shade = 255u) {
      for (int y = std::max(top, 0); y < std::min(bottom, static_cast<int>(height)); ++y) {
        for (int x = std::max(left, 0); x < std::min(right, static_cast<int>(width)); ++x) {
          std::uint8_t *texel = &pixels[(static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)) * 4u];
          if (coverage >= texel[3]) {
            texel[0] = texel[1] = texel[2] = shade;
            texel[3] = coverage;
          }
        }
      }
    }

    /**
     * One glyph in a cell of `size`: three-texel strokes with a soft edge, as a font draws them,
     * inside a dark outline of `outline` texels where one is asked for.
     */
    void glyph(const int x, const int y, const int size, const unsigned variant, const std::uint8_t shade = 255u, const int outline = 0) {
      const auto stroke = [&](const int left, const int top, const int right, const int bottom) {
        fill(left - outline - 1, top - outline - 1, right + outline + 1, bottom + outline + 1, 110u, outline ? 16u : shade);
        if (outline) {
          fill(left - outline, top - outline, right + outline, bottom + outline, 255u, 16u);
        }
        fill(left, top, right, bottom, 255u, shade);
      };
      const int third = size / 3;
      stroke(x + 2, y + 3, x + size - 2, y + 6);  // a bar across the top
      stroke(x + third + (variant % 3u) * 2, y + 3, x + third + (variant % 3u) * 2 + 3, y + size - 2);  // a stem
      if (variant % 2u == 0u) {
        stroke(x + 3, y + 2 * third, x + size - 3, y + 2 * third + 3);
      } else {
        stroke(x + size - 7, y + third, x + size - 4, y + size - 3);
      }
    }

    /** `lines` lines of `count` glyphs, as a dialogue box holds them. */
    void lines(const int left, const int top, const int lines, const int count, const int size, const std::uint8_t shade = 255u, const int outline = 0) {
      for (int line = 0; line < lines; ++line) {
        for (int index = 0; index < count; ++index) {
          glyph(left + index * (size + 2), top + line * (size + 12), size, static_cast<unsigned>(line * 7 + index), shade, outline);
        }
      }
    }

    [[nodiscard]] image::measure_t examine(const std::uint32_t budget = 65536u) const {
      return image::examine(pixels.data(), width * 4u, width, height, 4u, 3u, budget);
    }
  };

  TEST(TextHookImage, ALayerOfDialogueLinesIsText) {
    canvas_t layer(1600u, 900u);
    layer.lines(180, 650, 3, 40, 28);
    const auto measure = layer.examine();
    EXPECT_EQ(measure.verdict, image::verdict_e::text) << measure.covered_percent << ' ' << measure.solid_percent << ' ' << measure.narrow_percent;
    EXPECT_FALSE(measure.dense);
    // Looked at more coarsely, as memory that is slow to read is.
    EXPECT_EQ(layer.examine(16384u).verdict, image::verdict_e::text);
    // A whole screen of text, as a backlog shows, and large text, as a title is.
    canvas_t backlog(1600u, 900u);
    backlog.lines(100, 40, 20, 46, 28);
    EXPECT_EQ(backlog.examine().verdict, image::verdict_e::text);
    canvas_t title(1600u, 900u);
    title.lines(200, 300, 1, 12, 96);
    EXPECT_EQ(title.examine().verdict, image::verdict_e::text);
  }

  TEST(TextHookImage, TextMayBeOfAnyColoursAndMayCarryAShadowOrAnOutline) {
    canvas_t layer(1600u, 900u);
    // A soft shadow under every line joins its glyphs into one covered band.
    layer.fill(170, 640, 1400, 780, 70u, 0u);
    layer.lines(180, 650, 3, 40, 28);
    const auto shadowed = layer.examine();
    EXPECT_EQ(shadowed.verdict, image::verdict_e::text) << shadowed.covered_percent << ' ' << shadowed.solid_percent << ' ' << shadowed.narrow_percent;
    // A name in one colour, its line in another, a word picked out in a third.
    canvas_t coloured(1600u, 900u);
    coloured.lines(180, 600, 1, 6, 28, 200u);
    coloured.lines(180, 650, 2, 40, 28, 255u);
    coloured.lines(600, 730, 1, 8, 28, 90u);
    EXPECT_EQ(coloured.examine().verdict, image::verdict_e::text);
    // An outline joins the strokes of a glyph into one solid shape, and a heavy one the glyphs
    // of a line into a band: neither has a thin stroke left, and both end within a glyph's size.
    canvas_t outlined(1600u, 900u);
    outlined.lines(180, 600, 3, 30, 40, 255u, 3);
    const auto joined = outlined.examine();
    EXPECT_EQ(joined.verdict, image::verdict_e::text) << joined.covered_percent << ' ' << joined.solid_percent << ' ' << joined.narrow_percent;
    canvas_t heavy(1600u, 900u);
    heavy.lines(180, 600, 3, 30, 40, 255u, 6);
    const auto banded = heavy.examine();
    EXPECT_EQ(banded.verdict, image::verdict_e::text) << banded.covered_percent << ' ' << banded.solid_percent << ' ' << banded.narrow_percent;
    EXPECT_EQ(heavy.examine(16384u).verdict, image::verdict_e::text);
    // A layer made for a larger picture holds larger text.
    canvas_t large(3840u, 2160u);
    large.lines(400, 1500, 3, 28, 100, 255u, 8);
    const auto scaled = large.examine();
    EXPECT_EQ(scaled.verdict, image::verdict_e::text) << scaled.covered_percent << ' ' << scaled.solid_percent << ' ' << scaled.narrow_percent;
  }

  TEST(TextHookImage, APlateAVeilAFrameAndAPictureAreNotText) {
    // The box a dialogue is shown in, opaque: one long solid run on every row.
    canvas_t plate(1600u, 900u);
    plate.fill(50, 560, 1550, 890, 255u, 20u);
    EXPECT_EQ(plate.examine().verdict, image::verdict_e::other);
    // The same box as a veil: covered, and solid nowhere.
    canvas_t veil(1600u, 900u);
    veil.fill(50, 560, 1550, 890, 190u, 20u);
    EXPECT_EQ(veil.examine().verdict, image::verdict_e::other);
    // A veil inside a thin solid frame: hardly any of what is covered is solid.
    canvas_t frame(1600u, 900u);
    frame.fill(50, 560, 1550, 890, 190u, 20u);
    frame.fill(50, 560, 1550, 562, 255u);
    frame.fill(50, 888, 1550, 890, 255u);
    frame.fill(50, 560, 52, 890, 255u);
    frame.fill(1548, 560, 1550, 890, 255u);
    EXPECT_EQ(frame.examine().verdict, image::verdict_e::other);
    // A box and its text kept on one image: the mask of it would hold the box.
    canvas_t both(1600u, 900u);
    both.fill(50, 560, 1550, 890, 190u, 20u);
    both.lines(180, 650, 3, 40, 28);
    EXPECT_EQ(both.examine().verdict, image::verdict_e::other);
    // A sprite: a filled shape on nothing. A slender one too, with limbs as thin as a glyph.
    canvas_t sprite(1600u, 900u);
    sprite.fill(500, 100, 1100, 900, 255u, 180u);
    EXPECT_EQ(sprite.examine().verdict, image::verdict_e::other);
    canvas_t figure(1600u, 900u);
    figure.fill(760, 200, 900, 330, 255u, 200u);  // a head
    figure.fill(720, 330, 940, 900, 255u, 120u);  // a body
    figure.fill(680, 340, 720, 700, 255u, 200u);  // two arms
    figure.fill(940, 340, 980, 700, 255u, 200u);
    const auto slender = figure.examine();
    EXPECT_EQ(slender.verdict, image::verdict_e::other) << slender.narrow_percent;
    // The bars above and below a picture that is wider than the screen.
    canvas_t bars(1600u, 900u);
    bars.fill(0, 0, 1600, 110, 255u, 0u);
    bars.fill(0, 790, 1600, 900, 255u, 0u);
    EXPECT_EQ(bars.examine().verdict, image::verdict_e::other);
    // A plate on a layer made for a larger picture is still a plate.
    canvas_t large_plate(3840u, 2160u);
    large_plate.fill(200, 1500, 3600, 2100, 255u, 20u);
    EXPECT_EQ(large_plate.examine().verdict, image::verdict_e::other);
    // A picture: everything covered, which also ends a texture's time as a text texture at once.
    canvas_t picture(1600u, 900u);
    picture.fill(0, 0, 1600, 900, 255u, 128u);
    const auto measure = picture.examine();
    EXPECT_EQ(measure.verdict, image::verdict_e::other);
    EXPECT_TRUE(measure.dense);
    EXPECT_FALSE(plate.examine().dense);
  }

  TEST(TextHookImage, AFadedLayerIsStillTextAndTooLittleSaysNothing) {
    // A layer faded as a whole: its strokes are as solid as anything on it gets.
    canvas_t layer(1600u, 900u);
    layer.lines(180, 650, 3, 40, 28);
    for (std::size_t texel = 3u; texel < layer.pixels.size(); texel += 4u) {
      layer.pixels[texel] = static_cast<std::uint8_t>(layer.pixels[texel] * 6u / 10u);
    }
    const auto faded = layer.examine();
    EXPECT_EQ(faded.verdict, image::verdict_e::text) << faded.covered_percent << ' ' << faded.solid_percent << ' ' << faded.narrow_percent;
    // A veil faded the same way is solid all over, and as wide as ever.
    canvas_t veil(1600u, 900u);
    veil.fill(50, 560, 1550, 890, 120u, 20u);
    EXPECT_EQ(veil.examine().verdict, image::verdict_e::other);
    // Two glyphs on a large layer, looked at coarsely: too few texels to judge by. That is not
    // text, and no evidence against a texture that held text either.
    canvas_t few(1920u, 1080u);
    few.lines(900, 500, 1, 2, 28);
    const auto measure = few.examine(16384u);
    EXPECT_EQ(measure.verdict, image::verdict_e::other);
    EXPECT_TRUE(measure.faint);
    EXPECT_FALSE(layer.examine().faint);
  }

  // What is no larger than a glyph one way passes for text whatever it draws: a frame or a rule,
  // a narrow bar, a row of small buttons. That is the price of asking nothing about what the
  // shapes spell; whoever uses the verdict asks for more than one such image.
  TEST(TextHookImage, WhatIsAsSmallAsAGlyphPassesForText) {
    canvas_t frame(1600u, 900u);
    frame.fill(50, 560, 1550, 564, 255u);
    frame.fill(50, 886, 1550, 890, 255u);
    frame.fill(50, 560, 54, 890, 255u);
    frame.fill(1546, 560, 1550, 890, 255u);
    EXPECT_EQ(frame.examine().verdict, image::verdict_e::text);
    canvas_t bar(1600u, 900u);
    bar.fill(0, 0, 1600, 48, 255u, 0u);
    EXPECT_EQ(bar.examine().verdict, image::verdict_e::text);
    canvas_t buttons(1600u, 900u);
    for (int index = 0; index < 8; ++index) {
      buttons.fill(900 + index * 80, 840, 956 + index * 80, 896, 255u);
    }
    EXPECT_EQ(buttons.examine().verdict, image::verdict_e::text);
  }

  // An application that keeps its glyphs in one texture sets down more of them as the text goes
  // on. Full, the store is covered like no text layer and is as much a store of glyphs as before:
  // that is told apart from the verdict, for whoever knows the texture to hold text.
  TEST(TextHookImage, AStoreOfGlyphsThatHasFilledIsStillGlyphs) {
    canvas_t store(512u, 512u);
    const auto set_down = [&](const int from_row, const int to_row) {
      for (int row = from_row; row < to_row; ++row) {
        for (int column = 0; column < 12; ++column) {
          store.fill(column * 40 + 2, row * 40 + 2, column * 40 + 38, row * 40 + 38, 255u);
        }
      }
    };
    set_down(0, 4);
    const auto begun = store.examine();
    EXPECT_EQ(begun.verdict, image::verdict_e::text) << begun.covered_percent << ' ' << begun.solid_percent << ' ' << begun.narrow_percent;
    EXPECT_TRUE(begun.glyphs);
    set_down(4, 12);
    const auto full = store.examine();
    EXPECT_EQ(full.verdict, image::verdict_e::other) << full.covered_percent;
    EXPECT_TRUE(full.dense);
    EXPECT_TRUE(full.glyphs) << full.solid_percent << ' ' << full.narrow_percent;
    // A picture and a plate are neither, however much of the image they cover.
    canvas_t picture(512u, 512u);
    picture.fill(0, 0, 512, 512, 255u, 128u);
    EXPECT_FALSE(picture.examine().glyphs);
    canvas_t plate(1600u, 900u);
    plate.fill(50, 560, 1550, 890, 255u, 20u);
    EXPECT_FALSE(plate.examine().glyphs);
    // Nor are two glyphs looked at too coarsely to judge.
    canvas_t few(1920u, 1080u);
    few.lines(900, 500, 1, 2, 28);
    EXPECT_FALSE(few.examine(16384u).glyphs);
  }

  TEST(TextHookImage, AnEmptyLayerIsEmptyAndEveryImageHasItsOwnSum) {
    canvas_t layer(1600u, 900u);
    const auto empty = layer.examine();
    EXPECT_EQ(empty.verdict, image::verdict_e::empty);
    layer.lines(180, 650, 1, 20, 28);
    const auto first = layer.examine();
    // The next glyph of a line that is being revealed.
    layer.glyph(180 + 20 * 30, 650, 28, 3u);
    const auto second = layer.examine();
    EXPECT_NE(first.sum, empty.sum);
    EXPECT_NE(first.sum, second.sum);
    EXPECT_EQ(second.sum, layer.examine().sum);
  }

  TEST(TextHookImage, OneGlyphInItsCellAndOneChannelOfCoverageAreText) {
    // A cell of a glyph atlas: the glyph nearly fills it.
    canvas_t cell(32u, 32u);
    cell.glyph(1, 1, 30, 0u);
    cell.glyph(2, 2, 28, 1u);
    const auto measure = cell.examine();
    EXPECT_EQ(measure.verdict, image::verdict_e::text) << measure.covered_percent << ' ' << measure.solid_percent << ' ' << measure.narrow_percent;
    // Coverage alone, as an atlas of one channel holds it.
    canvas_t layer(512u, 128u);
    layer.lines(10, 20, 2, 16, 28);
    std::vector<std::uint8_t> coverage(static_cast<std::size_t>(layer.width) * layer.height);
    for (std::size_t texel = 0u; texel < coverage.size(); ++texel) {
      coverage[texel] = layer.pixels[texel * 4u + 3u];
    }
    EXPECT_EQ(image::examine(coverage.data(), layer.width, layer.width, layer.height, 1u, 0u, 65536u).verdict, image::verdict_e::text);
  }

  TEST(TextHookImage, WhatCannotBeReadIsNotText) {
    canvas_t layer(64u, 64u);
    layer.lines(2, 2, 1, 2, 28);
    EXPECT_EQ(image::examine(nullptr, 256u, 64u, 64u, 4u, 3u, 65536u).verdict, image::verdict_e::other);
    // A row shorter than its texels, no texels, no budget, coverage outside the texel.
    EXPECT_EQ(image::examine(layer.pixels.data(), 255u, 64u, 64u, 4u, 3u, 65536u).verdict, image::verdict_e::other);
    EXPECT_EQ(image::examine(layer.pixels.data(), 256u, 0u, 64u, 4u, 3u, 65536u).verdict, image::verdict_e::other);
    EXPECT_EQ(image::examine(layer.pixels.data(), 256u, 64u, 64u, 4u, 3u, 0u).verdict, image::verdict_e::other);
    EXPECT_EQ(image::examine(layer.pixels.data(), 256u, 64u, 64u, 4u, 4u, 65536u).verdict, image::verdict_e::other);
  }
}  // namespace
