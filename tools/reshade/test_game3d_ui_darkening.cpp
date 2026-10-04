// SPDX-License-Identifier: GPL-3.0-only
// Pin only UI (rule P2, fix 4; game3d_ui_darkening.h): the CPU reference on
// synthetic composites. Smooth dims unpin by opacity and by change set in
// sRGB and PQ; black UI with sharp edges (2-px glyph strokes, the edges of an
// opaque black panel, the inner part of a soft halo) stays pinned while the
// panel's interior and the halo's faint tail unpin; a grey tint over a grey
// scene reads as a darkening at the pixel but the tile gate keeps it, and a
// dim joined to it stays pinned with its region while a separate dim
// unpins; colour opacity evidence pins, a float source by the linear
// tolerance, and a colourless source unpins nothing; an isolated dark speck
// unpins; scRGB judges its residual relative to the presented peak; and the
// log text and its throttle.
#include "game3d_ui_darkening.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace sunshine_game3d;
  namespace dk = ui_darkening;
  using dk::rgb;

  void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
  }

  constexpr std::uint32_t W = 128, H = 128;
  constexpr float t8 = 2.f / 255.f, t10 = 4.f / 1023.f;

  std::size_t at(std::uint32_t x, std::uint32_t y) { return static_cast<std::size_t>(y) * W + x; }

  // A deterministic texture in [lo, hi] per channel.
  struct lcg {
    std::uint32_t state = 12345u;
    float next() {
      state = state * 1664525u + 1013904223u;
      return static_cast<float>(state >> 8) / 16777216.f;
    }
  };
  std::vector<rgb> texture(float lo, float hi, bool grey = false) {
    lcg random;
    std::vector<rgb> image(static_cast<std::size_t>(W) * H);
    for (auto &p : image) {
      for (auto &c : p) c = lo + (hi - lo) * random.next();
      if (grey) p[1] = p[2] = p[0];
    }
    return image;
  }
  float quantize(float value, float levels) { return std::round(std::clamp(value, 0.f, 1.f) * levels) / levels; }
  rgb quantize(const rgb &c, float levels) {
    return {quantize(c[0], levels), quantize(c[1], levels), quantize(c[2], levels)};
  }
  std::vector<rgb> quantized(std::vector<rgb> image, float levels) {
    for (auto &p : image) p = quantize(p, levels);
    return image;
  }
  // A PQ code of a linear colour (1 = 10000 nits).
  rgb pq_code(const rgb &linear) {
    return {dk::pq::inverse_eotf(linear[0]), dk::pq::inverse_eotf(linear[1]), dk::pq::inverse_eotf(linear[2])};
  }
  rgb pq_linear(const rgb &code) { return {dk::pq::eotf(code[0]), dk::pq::eotf(code[1]), dk::pq::eotf(code[2])}; }

  // One opaque colour pixel in the corner, far from every shape: an opacity
  // source without any colour unpins nothing (colourless).
  void add_colour_pixel(std::vector<rgb> &colour, std::vector<float> &alpha) {
    colour[at(W - 1u, H - 1u)] = {.9f, .5f, .1f};
    alpha[at(W - 1u, H - 1u)] = 1.f;
  }

  // The smooth band: opacity 0 at row 32 rising to 0.9 at the last row,
  // about 0.0095 per row.
  float band(std::uint32_t y) { return y < 32u ? 0.f : 0.9f * static_cast<float>(y - 32u) / static_cast<float>(H - 33u); }

  void dim_band_unpins() {
    // Opacity: a black premultiplied band (a straight black tag reads the
    // same) and one colour pixel above it, so that the source carries colour.
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size());
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) alpha[at(x, y)] = band(y);
    colour[at(5, 5)] = {.9f, .5f, .1f};
    alpha[at(5, 5)] = 1.f;
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    require(o.n.covered == static_cast<std::uint64_t>(W) * (H - 33u) + 1u && o.n.darkening + 1u == o.n.covered &&
        o.n.unpinned == o.n.darkening && o.n.kept == 0u,
      "An opacity dim band must unpin every pixel: unpinned " + std::to_string(o.n.unpinned) + " of " +
        std::to_string(o.n.covered));
    // Change sets: the same band composited over a textured scene, in sRGB
    // (8-bit, encoded blend) as a declared pair (k 1) and as the pre-UI
    // change set (k 8, 3x3 rule), and in PQ (10-bit, blended in linear).
    const auto scene8 = quantized(texture(.3f, .8f), 255.f);
    std::vector<rgb> dimmed8(scene8.size());
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) {
        const auto &l = scene8[at(x, y)];
        const float s = 1.f - band(y);
        dimmed8[at(x, y)] = quantize(rgb{s * l[0], s * l[1], s * l[2]}, 255.f);
      }
    for (const float k : {1.f, 8.f}) {
      const auto mask = dk::change_set_mask(W, H, dimmed8, scene8, t8, k, dk::color_space::srgb, k > 1.f);
      const auto c = dk::reference_change_set(W, H, dimmed8, scene8, mask, t8, k, dk::color_space::srgb);
      require(c.n.covered > 9000u && c.n.darkening == c.n.covered && c.n.unpinned * 100u >= c.n.covered * 99u &&
          c.n.tiles_colour == 0u,
        "An sRGB change-set dim band (k " + std::to_string(k) + ") must unpin: unpinned " + std::to_string(c.n.unpinned) +
          " of " + std::to_string(c.n.covered) + ", darkening " + std::to_string(c.n.darkening) + ", colour tiles " +
          std::to_string(c.n.tiles_colour));
      std::printf("PASS dim band sRGB change set k %.0f: %llu of %llu covered unpinned, %llu darkening tiles\n", k,
        static_cast<unsigned long long>(c.n.unpinned), static_cast<unsigned long long>(c.n.covered),
        static_cast<unsigned long long>(c.n.tiles_darkening));
    }
    auto scene_pq = texture(.002f, .02f);
    for (auto &p : scene_pq) p = quantize(pq_code(p), 1023.f);
    std::vector<rgb> dimmed_pq(scene_pq.size());
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) {
        const auto l = pq_linear(scene_pq[at(x, y)]);
        const float s = 1.f - band(y);
        dimmed_pq[at(x, y)] = quantize(pq_code(rgb{s * l[0], s * l[1], s * l[2]}), 1023.f);
      }
    const auto mask = dk::change_set_mask(W, H, dimmed_pq, scene_pq, t10, 1.f, dk::color_space::pq, false);
    const auto c = dk::reference_change_set(W, H, dimmed_pq, scene_pq, mask, t10, 1.f, dk::color_space::pq);
    require(c.n.covered > 9000u && c.n.darkening == c.n.covered && c.n.unpinned * 100u >= c.n.covered * 99u &&
        c.n.tiles_colour == 0u,
      "A PQ change-set dim band blended in linear must unpin: unpinned " + std::to_string(c.n.unpinned) + " of " +
        std::to_string(c.n.covered) + ", darkening " + std::to_string(c.n.darkening));
    std::printf("PASS dim band: opacity %llu of %llu unpinned; PQ change set (linear blend) %llu of %llu unpinned\n",
      static_cast<unsigned long long>(o.n.unpinned), static_cast<unsigned long long>(o.n.covered),
      static_cast<unsigned long long>(c.n.unpinned), static_cast<unsigned long long>(c.n.covered));
  }

  // 2-px black strokes: a box outline and a cross.
  bool stroke(std::uint32_t x, std::uint32_t y) {
    const bool box = x >= 20u && x < 60u && y >= 20u && y < 50u && (x < 22u || x >= 58u || y < 22u || y >= 48u);
    const bool cross = (x >= 80u && x < 82u && y >= 20u && y < 100u) || (y >= 60u && y < 62u && x >= 70u && x < 110u);
    return box || cross;
  }

  void glyph_strokes_kept() {
    // Every stroke pixel beside the background sits on the opacity step and
    // stays pinned; only the 2x2 junctions (the box's corners and the cross's
    // centre), whose four neighbours are all stroke, unpin like a panel's
    // interior.
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size(), 0.f);
    const auto scene = quantized(texture(.2f, .9f), 255.f);
    auto presented = scene;
    std::uint64_t strokes = 0, edges = 0;
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x)
        if (stroke(x, y)) {
          alpha[at(x, y)] = 1.f;
          presented[at(x, y)] = rgb{};
          ++strokes;
          edges += !stroke(x - 1u, y) || !stroke(x + 1u, y) || !stroke(x, y - 1u) || !stroke(x, y + 1u) ? 1u : 0u;
        }
    require(strokes - edges == 8u, "The strokes must have eight junction pixels");
    add_colour_pixel(colour, alpha);
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    require(o.n.darkening == strokes && o.n.kept == edges && o.n.unpinned == strokes - edges,
      "Opaque black 2-px strokes must stay pinned (opacity): kept " + std::to_string(o.n.kept) + " of " +
        std::to_string(edges));
    const auto mask = dk::change_set_mask(W, H, presented, scene, t8, 1.f, dk::color_space::srgb, false);
    const auto c = dk::reference_change_set(W, H, presented, scene, mask, t8, 1.f, dk::color_space::srgb);
    // As a change set the junctions also need their region to unpin. Kept
    // pixels join no region, so each junction's tile is a region of its own:
    // the box's lower corners lie in tiles with only 24 darkening pixels (no
    // verdict, so no proof), and those two stay pinned.
    bool edges_kept = true;
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x)
        if (stroke(x, y) && (!stroke(x - 1u, y) || !stroke(x + 1u, y) || !stroke(x, y - 1u) || !stroke(x, y + 1u)))
          edges_kept = edges_kept && c.classes[at(x, y)] == dk::pixel_class::kept;
    const auto corner = (48u / dk::tile) * c.tiles_x + 21u / dk::tile;
    require(c.n.darkening == strokes && edges_kept && c.n.kept == edges + 2u && c.n.unpinned == strokes - edges - 2u &&
        c.verdicts[corner] == dk::verdict::none && c.component[corner] >= 0 && !c.region[corner] &&
        c.classes[at(21, 48)] == dk::pixel_class::kept && c.classes[at(58, 48)] == dk::pixel_class::kept,
      "Opaque black 2-px strokes must stay pinned (change set): kept " + std::to_string(c.n.kept) + " of " +
        std::to_string(edges) + " edge pixels and 2 junctions in regions without a proof");
    std::printf("PASS black 2-px strokes: %llu edge pixels of %llu kept by opacity and by change set; junctions "
                "unpinned (opacity %llu, change set %llu in proven regions)\n",
      static_cast<unsigned long long>(edges), static_cast<unsigned long long>(strokes),
      static_cast<unsigned long long>(o.n.unpinned), static_cast<unsigned long long>(c.n.unpinned));
  }

  void black_panel_edges_kept() {
    // A 40x60 opaque black panel: its one-pixel rim sits on the opacity step
    // and stays pinned; the interior unpins (the pin band protects it near
    // the rim).
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size(), 0.f);
    for (std::uint32_t y = 40; y != 100; ++y)
      for (std::uint32_t x = 40; x != 80; ++x) alpha[at(x, y)] = 1.f;
    add_colour_pixel(colour, alpha);
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    const std::uint64_t rim = 2u * 40u + 2u * 58u;
    require(o.n.covered == 2401u && o.n.kept == rim && o.n.unpinned == 2400u - rim,
      "A black panel must keep exactly its rim: kept " + std::to_string(o.n.kept) + ", unpinned " +
        std::to_string(o.n.unpinned));
    require(o.classes[at(40, 70)] == dk::pixel_class::kept && o.classes[at(79, 70)] == dk::pixel_class::kept &&
        o.classes[at(60, 40)] == dk::pixel_class::kept && o.classes[at(40, 40)] == dk::pixel_class::kept &&
        o.classes[at(41, 70)] == dk::pixel_class::unpinned && o.classes[at(60, 70)] == dk::pixel_class::unpinned,
      "The panel's rim must be kept and its interior unpinned");
    std::printf("PASS black panel: rim %llu kept, interior %llu unpinned\n", static_cast<unsigned long long>(o.n.kept),
      static_cast<unsigned long long>(o.n.unpinned));
  }

  void soft_halo_tail_unpins() {
    // A 10x10 colour icon with a black halo around it whose opacity falls
    // from 0.25 by 0.6 per pixel over 8 pixels (0.25, 0.15, 0.09, 0.054,
    // 0.032, 0.019, 0.012, 0.007): its steps reach 1/64 down to the fifth
    // ring, so the inner halo stays pinned and the faint tail unpins. The
    // fifth ring's four corners step only outward (0.013) and unpin too.
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size(), 0.f);
    std::vector<int> ring(colour.size(), -1);
    std::vector<bool> corner(colour.size(), false);
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) {
        const int dx = std::max({59 - static_cast<int>(x), static_cast<int>(x) - 68, 0});
        const int dy = std::max({59 - static_cast<int>(y), static_cast<int>(y) - 68, 0});
        const int d = std::max(dx, dy);
        ring[at(x, y)] = d;
        corner[at(x, y)] = dx == dy;
        if (d == 0) {
          colour[at(x, y)] = {.9f, .5f, .1f};
          alpha[at(x, y)] = 1.f;
        } else if (d <= 8) {
          alpha[at(x, y)] = .25f * std::pow(.6f, static_cast<float>(d - 1));
        }
      }
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    for (std::size_t i = 0; i != ring.size(); ++i) {
      const int d = ring[i];
      if (d == 0) require(o.classes[i] == dk::pixel_class::colour, "The icon must be colour UI");
      else if (d < 5 || (d == 5 && !corner[i]))
        require(o.classes[i] == dk::pixel_class::kept, "Halo ring " + std::to_string(d) + " must stay pinned");
      else if (d <= 8)
        require(o.classes[i] == dk::pixel_class::unpinned && alpha[i] < .033f,
          "Halo ring " + std::to_string(d) + " (alpha below 0.033) must unpin");
    }
    std::printf("PASS soft halo: rings 1-5 (alpha >= 0.032) kept %llu, rings 6-8 (alpha <= 0.019) and the fifth ring's "
                "corners unpinned %llu\n",
      static_cast<unsigned long long>(o.n.kept), static_cast<unsigned long long>(o.n.unpinned));
  }

  void grey_tint_stays_pinned() {
    // A grey UI tint U 0.3 at a 0.5 over a smooth grey scene: every pixel
    // reads P = s L, a darkening, and the opacity varies too smoothly for a
    // step, so only the tile gate keeps it pinned.
    std::vector<rgb> scene(static_cast<std::size_t>(W) * H), presented(scene.size());
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) {
        const float l = quantize(.4f + .5f * static_cast<float>(x) / static_cast<float>(W - 1u), 255.f);
        scene[at(x, y)] = {l, l, l};
        const bool tinted = x >= 16u && x < 112u && y >= 16u && y < 112u;
        const float p = tinted ? quantize(.5f * l + .15f, 255.f) : l;
        presented[at(x, y)] = {p, p, p};
      }
    const auto mask = dk::change_set_mask(W, H, presented, scene, t8, 1.f, dk::color_space::srgb, false);
    const auto c = dk::reference_change_set(W, H, presented, scene, mask, t8, 1.f, dk::color_space::srgb);
    std::uint64_t structured = 0;
    for (const auto s : c.structure) structured += s;
    require(c.n.covered > 8000u && c.n.darkening * 10u >= c.n.covered * 9u && structured * 10u < c.n.darkening,
      "The tint must read as an unstructured darkening at the pixel: darkening " + std::to_string(c.n.darkening) +
        " of " + std::to_string(c.n.covered) + ", structured " + std::to_string(structured));
    require(c.n.tiles_colour >= 30u && c.n.tiles_darkening == 0u && c.n.unpinned == 0u,
      "The tile gate must call the tint colour and unpin nothing: colour tiles " + std::to_string(c.n.tiles_colour) +
        ", darkening tiles " + std::to_string(c.n.tiles_darkening) + ", unpinned " + std::to_string(c.n.unpinned));
    // The same tint as opacity evidence (premultiplied colour 0.15) is colour.
    std::vector<rgb> colour(scene.size(), rgb{});
    std::vector<float> alpha(scene.size(), 0.f);
    for (std::uint32_t y = 16; y != 112; ++y)
      for (std::uint32_t x = 16; x != 112; ++x) {
        colour[at(x, y)] = {.15f, .15f, .15f};
        alpha[at(x, y)] = .5f;
      }
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    require(o.n.covered == 96u * 96u && o.n.darkening == 0u && o.n.unpinned == 0u &&
        o.classes[at(50, 50)] == dk::pixel_class::colour,
      "A grey tint with colour 0.15 must be colour UI");
    std::printf("PASS grey tint: %llu of %llu change-set pixels darkening, %llu colour tiles, nothing unpinned; opacity "
                "colour 0.15 pins\n",
      static_cast<unsigned long long>(c.n.darkening), static_cast<unsigned long long>(c.n.covered),
      static_cast<unsigned long long>(c.n.tiles_colour));
  }

  void joined_dim_stays_pinned() {
    // A grey tint (tile rows 1-3) over a grey scene with a pure dim to 0.5
    // joined below it (tile rows 4-8) and a separate dim at the left edge
    // (tile column 0, two tiles away). The tint's colour verdicts ring tile
    // rows 0-4 only, so a per-tile gate would unpin tile rows 5-8 of the
    // joined dim beside the pinned tint (a 16 px block edge inside one
    // widget); as one region with the tint it stays pinned, while the
    // separate dim unpins.
    constexpr std::uint32_t h = 160;
    std::vector<rgb> scene(static_cast<std::size_t>(W) * h), presented(scene.size());
    for (std::uint32_t y = 0; y != h; ++y)
      for (std::uint32_t x = 0; x != W; ++x) {
        const float l = quantize(.4f + .5f * static_cast<float>((x * 7u + y * 3u) % 29u) / 28.f, 255.f);
        scene[static_cast<std::size_t>(y) * W + x] = {l, l, l};
        const bool tinted = x >= 48u && x < 112u && y >= 16u && y < 64u;
        const bool joined = x >= 48u && x < 112u && y >= 64u && y < 144u;
        const bool separate = x < 16u && y >= 16u && y < 144u;
        const float p = tinted ? quantize(.5f * l + .15f, 255.f) : joined || separate ? quantize(.5f * l, 255.f) : l;
        presented[static_cast<std::size_t>(y) * W + x] = {p, p, p};
      }
    const auto mask = dk::change_set_mask(W, h, presented, scene, t8, 1.f, dk::color_space::srgb, false);
    const auto c = dk::reference_change_set(W, h, presented, scene, mask, t8, 1.f, dk::color_space::srgb);
    const auto tile_at = [&](std::uint32_t x, std::uint32_t y) {
      return static_cast<std::size_t>(y / dk::tile) * c.tiles_x + x / dk::tile;
    };
    const auto cls = [&](std::uint32_t x, std::uint32_t y) { return c.classes[static_cast<std::size_t>(y) * W + x]; };
    // Tile row 6 (rows 96-111) of the joined dim proves darkening beyond the
    // tint's ring (tile rows 0-4), yet shares the tint's region.
    require(c.verdicts[tile_at(80, 100)] == dk::verdict::darkening && !c.ring[tile_at(80, 100)] &&
        c.component[tile_at(80, 100)] == c.component[tile_at(80, 30)] && !c.region[tile_at(80, 100)] &&
        cls(80, 100) == dk::pixel_class::kept && cls(80, 30) != dk::pixel_class::unpinned,
      "A proven dim tile joined to a tinted panel must stay pinned with its region");
    require(c.region[tile_at(8, 80)] && cls(8, 80) == dk::pixel_class::unpinned && c.n.regions == 2u &&
        c.n.regions_unpinned == 1u,
      "A separate dim must unpin as its own region: regions " + std::to_string(c.n.regions) + ", unpinning " +
        std::to_string(c.n.regions_unpinned));
    std::printf("PASS joined dim: %llu regions, %llu unpins; the dim joined to the tint stays pinned (%llu unpinned px, "
                "all in the separate dim)\n",
      static_cast<unsigned long long>(c.n.regions), static_cast<unsigned long long>(c.n.regions_unpinned),
      static_cast<unsigned long long>(c.n.unpinned));
  }

  void colourless_and_linear_opacity() {
    // A layer with real alpha and zero colour (coverage only) carries no
    // colour evidence: nothing unpins, every darkening pixel is kept.
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size());
    for (std::uint32_t y = 0; y != H; ++y)
      for (std::uint32_t x = 0; x != W; ++x) alpha[at(x, y)] = band(y);
    auto o = dk::reference_opacity(W, H, colour, alpha);
    require(o.colourless && o.n.unpinned == 0u && o.n.kept == o.n.darkening && o.n.kept > 0u,
      "A colourless opacity source must unpin nothing");
    // One colour pixel anywhere makes the source live.
    colour[at(5, 5)] = {.5f, .2f, .1f};
    alpha[at(5, 5)] = 1.f;
    o = dk::reference_opacity(W, H, colour, alpha);
    require(!o.colourless && o.n.unpinned > 0u, "One colour pixel must make the opacity source live");
    // A charcoal #303030 panel at alpha 0.5: 24/255 in an 8-bit sRGB layer
    // (colour), 0.0148 linear in a float layer, below 4/255 but above the
    // linear tolerance, so colour in both; the code tolerance on linear
    // values would call it a darkening.
    const float charcoal = static_cast<float>(std::pow((48. / 255. + .055) / 1.055, 2.4)) * .5f;
    const auto sdr = dk::opacity(rgb{24.f / 255.f, 24.f / 255.f, 24.f / 255.f}, .5f, false);
    const auto hdr = dk::opacity(rgb{charcoal, charcoal, charcoal}, .5f, true);
    const auto wrong = dk::opacity(rgb{charcoal, charcoal, charcoal}, .5f, false);
    require(sdr.colour_raw && hdr.colour_raw && wrong.dark && std::abs(dk::opacity_tolerance_linear - .001214f) < 1e-6f,
      "A charcoal panel must be colour in SDR and in a float layer");
    std::printf("PASS colourless opacity source unpins nothing; charcoal #303030 at 0.5 is colour in SDR (24/255) and "
                "in a float layer (%.4f linear > %.5f)\n", charcoal, dk::opacity_tolerance_linear);
  }

  void isolated_speck_unpins() {
    std::vector<rgb> colour(static_cast<std::size_t>(W) * H, rgb{});
    std::vector<float> alpha(colour.size(), 0.f);
    alpha[at(64, 64)] = .5f;
    add_colour_pixel(colour, alpha);
    const auto o = dk::reference_opacity(W, H, colour, alpha);
    require(o.n.covered == 2u && o.n.unpinned == 1u && o.structure[at(64, 64)] == 1u && o.kept[at(64, 64)] == 0u,
      "An isolated dark speck is on a step but alone in its 3x3 window, so it unpins");
    std::puts("PASS isolated speck: on a step, alone in its 3x3 window, unpinned");
  }

  void pq_linear_dim() {
    // A coloured PQ pixel dimmed to 40% in linear light is a darkening; the
    // same pixel carrying its own colour is not.
    const rgb linear{.008f, .005f, .002f};
    const auto pre = quantize(pq_code(linear), 1023.f);
    const auto l = pq_linear(pre);
    const auto dimmed = quantize(pq_code(rgb{.4f * l[0], .4f * l[1], .4f * l[2]}), 1023.f);
    const auto e = dk::change_set(dimmed, pre, t10, 1.f, true, dk::color_space::pq);
    require(e.dark && std::abs(e.a - .6f) < .02f, "A PQ dim blended in linear must be a darkening of opacity 0.6, a " +
      std::to_string(e.a));
    const auto tinted = quantize(pq_code(rgb{.4f * l[0] + .002f, .4f * l[1], .4f * l[2]}), 1023.f);
    const auto f = dk::change_set(tinted, pre, t10, 1.f, true, dk::color_space::pq);
    require(!f.dark && f.colour_raw, "A PQ dim with its own red must be colour");
    // Brightening is never a darkening, even within the residual bound.
    const auto g = dk::change_set(rgb{.5f, .5f, .5f}, rgb{.49f, .49f, .49f}, t8, 1.f, true, dk::color_space::srgb);
    require(!g.dark && g.colour_raw, "A brightened pixel must not be a darkening");
    // Yet it fits within the residual bound, so a declared pair's tile
    // regression pools it (a tint's lifted pixels count against the tile).
    require(g.fits, "A slightly brightened pixel of the mask must still fit for the tile regression");
    std::printf("PASS PQ linear dim: darkening at a %.3f; an added red and a brightening are colour\n", e.a);
  }

  void scrgb_relative() {
    // scRGB judges the residual relative to the presented peak above 1: a
    // 0.05 residual on a peak of 2 is 0.025, within 4 t (0.031); the same
    // residual on a peak below 1 is not.
    const rgb bright_pre{4.f, 3.f, 2.f}, bright{2.05f, 1.45f, 1.f};
    const auto e = dk::change_set(bright, bright_pre, t8, 1.f, true, dk::color_space::scrgb);
    require(e.dark, "A bright scRGB dim with a relative residual of 0.025 must be a darkening");
    const auto fit = dk::fit_darkening(bright, bright_pre, dk::color_space::scrgb);
    require(fit.residual > 0.f && fit.residual * std::max(1.f, dk::peak(bright)) > 4.f * t8,
      "The bright case's absolute residual must exceed 4 t");
    const rgb dim_pre{.8f, .6f, .4f}, dim{.48f, .28f, .2f};
    const auto f = dk::change_set(dim, dim_pre, t8, 1.f, true, dk::color_space::scrgb);
    require(!f.dark, "An scRGB residual near 0.05 below a peak of 1 must not be a darkening");
    std::puts("PASS scRGB relative residual: 0.05 over a peak of 2.05 darkens, below a peak of 1 does not");
  }

  void tile_verdicts() {
    // Fewer than 32 darkening pixels, or a flat tile, give no verdict.
    dk::tile_sums sums;
    for (int i = 0; i != 31; ++i) sums.add(rgb{.2f, .2f, .2f}, rgb{.4f + .01f * i, .4f, .4f}, dk::color_space::srgb);
    require(dk::tile_verdict(sums, t8, dk::color_space::srgb) == dk::verdict::none, "31 pixels must give no verdict");
    sums.add(rgb{.2f, .2f, .2f}, rgb{.7f, .4f, .4f}, dk::color_space::srgb);
    require(dk::tile_verdict(sums, t8, dk::color_space::srgb) != dk::verdict::none, "32 pixels must give a verdict");
    dk::tile_sums flat;
    for (int i = 0; i != 64; ++i) flat.add(rgb{.2f, .2f, .2f}, rgb{.4f, .4f, .4f}, dk::color_space::srgb);
    require(dk::tile_verdict(flat, t8, dk::color_space::srgb) == dk::verdict::none, "A flat grey tile must give no verdict");
    dk::tile_sums dim;
    for (int i = 0; i != 64; ++i) {
      const float l = .3f + .005f * i;
      dim.add(rgb{.5f * l, .5f * l, .5f * l}, rgb{l, l, l}, dk::color_space::srgb);
    }
    require(dk::tile_verdict(dim, t8, dk::color_space::srgb) == dk::verdict::darkening && dk::tile_effect(dim, t8, 1u) < .01,
      "A pure scale must be a darkening tile");
    std::puts("PASS tile verdicts: none below 32 pixels or without spread, darkening for a pure scale");
  }

  void sources_and_log_text() {
    for (const std::uint32_t source : {2u, 5u, 10u, 12u}) require(dk::eligible(source), "Source must be eligible");
    for (const std::uint32_t source : {0u, 1u, 3u, 4u, 6u, 8u, 11u})
      require(!dk::eligible(source), "Source must not be eligible");
    require(dk::opacity_source(2u) && dk::opacity_source(10u) && dk::change_set_source(5u) && dk::change_set_source(12u) &&
        !dk::opacity_source(5u) && !dk::change_set_source(10u),
      "The opacity and change-set sources");
    const auto text = dk::log_text({10u, false, 1365596u, 1065506u, 78689u});
    require(text == "Sunshine UI darkening: source=10 UIPinOnlyUI=0 covered=1365596 unpinned=1065506 kept=78689 colourless=0",
      "Unexpected log text: " + text);
    require(dk::log_text({12u, true, 5u, 0u, 3u}) ==
        "Sunshine UI darkening: source=12 UIPinOnlyUI=1 covered=5 unpinned=0 kept=3 colourless=0",
      "Unexpected log text with the switch on");
    const auto colourless = dk::sample_of(10u, true, 9u, 0u, 7u | ui_detection::darkening::colourless);
    require(colourless.kept == 7u && colourless.colourless &&
        dk::log_text(colourless) == "Sunshine UI darkening: source=10 UIPinOnlyUI=1 covered=9 unpinned=0 kept=7 colourless=1",
      "Word 63's colourless bit must be split from the kept count");
    // The line logs on a change (source, switch, unpinning or not, the
    // colourless bit), else at most once a second.
    dk::log_state log;
    const dk::sample a{10u, false, 100u, 5u, 3u}, b{10u, false, 100u, 9u, 1u};
    require(dk::log_due(log, a, 1000) && !dk::log_due(log, b, 1999) && dk::log_due(log, b, 2000),
      "The darkening line must throttle to once a second");
    require(dk::log_due(log, dk::sample{10u, false, 100u, 0u, 8u}, 2001) && dk::log_due(log, dk::sample{5u, false, 100u, 0u, 8u}, 2002) &&
        dk::log_due(log, dk::sample{5u, true, 100u, 0u, 8u}, 2003) && !dk::log_due(log, dk::sample{5u, true, 90u, 0u, 2u}, 2004),
      "The darkening line must log at once when its source, switch or unpinning changes");
    std::puts("PASS eligible sources 2, 5, 10, 12, the log text with the colourless bit and its throttle");
  }
}

int main() {
  try {
    dim_band_unpins();
    glyph_strokes_kept();
    black_panel_edges_kept();
    soft_halo_tail_unpins();
    grey_tint_stays_pinned();
    isolated_speck_unpins();
    pq_linear_dim();
    scrgb_relative();
    tile_verdicts();
    joined_dim_stays_pinned();
    colourless_and_linear_opacity();
    sources_and_log_text();
    std::puts("Pin only UI (P2) darkening reference: 12 groups passed");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
