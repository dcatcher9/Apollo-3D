// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Pin only UI (UI decision framework rule P2, fix 4; docs/reshade-sbs.md, UI
// decision framework): the CPU reference of the darkening rule that the mask
// pass (M7) applies, which ui_detection_replay and the GPU contract tests
// mirror, and the log text of its samples (game3d_log_report.py parses it).
//
// Ruling (10-04): only UI pins. A pure darkening, where the UI contributes no
// colour of its own and only scales the scene down (dims, vignettes, fades,
// menu backdrops), is not UI in SDR or HDR, so the scene under it keeps its
// depth. For a composite P = L (1 - a) + U a the UI's own colour is the
// premultiplied U a. Opacity evidence carries it: the offscreen UI layer's
// premultiplied colour (source 10) and the UI color tag's colour (source 2;
// a straight tag only overstates U a, so it errs toward pinning). Change-set
// evidence (the HUD-less pair, source 5, and the pre-UI change set, source
// 12) gives it as the residual of the best darkening fit P ~ s L in the fit
// space (linear light for PQ, the encoded values otherwise), with opacity
// a = 1 - s. Alpha-only sources (UIAlpha 1, Backbuffer 3, current alpha 4)
// carry no colour and are not eligible.
//
// Black UI (glyph outlines, drop shadows, black icon parts, opaque black
// panels) is also a darkening at the pixel, so structure separates it: a
// darkening pixel stays pinned (kept) when it sits on a sharp 4-neighbour
// opacity step (beyond both pixels' opacity uncertainty sigma) or touches
// raw colour UI, and at least min_neighbourhood such pixels share its 3x3
// window (change_set::min_neighbourhood, which removes isolated specks and
// keeps one-pixel lines). Only edges need it: the pin band already clamps
// the horizontal-only warp to 0.5 px per px around a pinned texel, so the
// interior of a black panel near its kept edges cannot be torn into thin
// strokes, and no dilation by the parallax limit is needed.
//
// Change sets also need the tile gate: one pixel cannot tell a pure dim from
// a grey tint over a grey scene (both read P = s L). Over each 16x16 tile
// (from the frame origin) the pooled channels of the mask's fitting pixels
// (residual within the bound, darker or not) fit f(P) = m f(L) + C; C is the
// tile's own colour U a, judged by its effect at the tile's darkened level in
// units of the pair threshold t. The gate judges whole darkening regions,
// not tiles: a region is a set of 8-connected tiles that hold darkening
// pixels structure does not keep (kept pixels are pinned anyway, and
// separate dims do not merge through the outlines between them), and
// its darkening pixels unpin only when one of its tiles is proven (a
// darkening verdict and no colour verdict in its 3x3 tile ring) and none of
// its tiles has a colour verdict in its ring. One translucent widget is one
// region, so it is either all dim or all UI, without 16 px blocks of each
// that the horizontal-only pin band could not join (Hogwarts Legacy's
// minimap disc beside its map lines); a tile without a verdict (fewer than
// tile_min_pixels fitting pixels, or no spread) takes its region's.
// Opacity sources carry their colour directly and skip the gate, but one
// whose colour is zero on every pixel of the frame (an engine that tags or
// renders only coverage) carries no colour evidence and unpins nothing.
// The regression keeps the fitting pixels a tint lifts to or just above the
// scene (s_raw >= 1, no darkening themselves): leaving them out biases a
// tinted tile toward a darkening verdict (Hogwarts Legacy's minigame panel:
// 11% more unpinned pixels). Only a declared pair (k 1) has such pixels; at
// the pre-UI change set's k 8 a changed pixel with s = 1 has a residual
// beyond 8 t, so its fitting pixels are exactly its darkening pixels.
//
// Every class here comes from the source's own evidence: nothing changes S1,
// acceptance, H1/H2, T1 or any decision word 0-61; the switch (UIPinOnlyUI)
// only zeroes the pin weight of unpinned pixels. No ReShade dependency.
//
// Precision: the per-pixel evidence is computed in float, as the shader
// does; the tile regression accumulates in double here, while the GPU
// reduces in float. Parity is checked within a tolerance band (tiles whose
// effect lies near tile_effect_scale, pixels whose residual or step lies at
// its bound), not bit for bit.
#include "game3d_ui_detection_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sunshine_game3d::ui_darkening {
  namespace contract = ui_detection::darkening;
  // Opacity sources: darkening when the UI's own colour max |U a| is at most
  // 4/255, V1's premultiplied slack. On Stellar Blade's HDR Equipment layer
  // 53.1% of UI pixels carry colour <= 2/255, 83.8% <= 4/255 and 83.9%
  // <= 8/255: 4/255 is the knee. The tolerance is in sRGB code, as UNORM
  // layers and tags store their colour; a float layer (stored_hdr_headroom)
  // or tag (rules::tag_linear) stores linear scRGB, so it compares with the
  // sRGB EOTF of 4/255 (about 0.0012), and an SDR composite and its HDR
  // counterpart get the same verdict (a charcoal #303030 panel at alpha 0.5
  // is colour in both).
  inline constexpr float opacity_tolerance = 4.f / 255.f, opacity_tolerance_linear = opacity_tolerance / 12.92f;
  constexpr float opacity_tolerance_for(bool linear) { return linear ? opacity_tolerance_linear : opacity_tolerance; }
  // Change sets: darkening when the fit residual R is at most 4 t (t the
  // pair threshold, b2 word 1), the inferred pair's unchanged bound
  // k t / 2 at k = change_set::inferred_scale (8), reused for declared pairs.
  // Stellar Blade's SDR Equipment band (dump 033): residual median 3.05/255,
  // p90 9.5/255 at t = 2/255, 96.7% of the band within 4 t against 66% at
  // 1 t; the exact pair 478's noise: 94% of changed pixels within 4 t.
  inline constexpr float change_set_tolerance_scale = 4.f;
  // A sharp opacity step: |a(x) - a(n)| - sigma(x) - sigma(n) >= 1/64. The
  // largest per-pixel step inside dims: 033's band 0.0033/px, Stellar Blade
  // HDR's band q90 0.0078, Witcher 3's sign-wheel backdrop q99 0.0039; HUD
  // halos step 0.02-0.1/px and glyph edges >= 0.25. On Stellar Blade's HDR
  // HUD layers 1/8 would remove 32-37% of the pinned weight (halos), 1/64
  // removes 3-4% (halo tails).
  inline constexpr float step = 1.f / 64.f;
  // The tile gate (change sets only): 16x16 tiles; a verdict needs at least
  // 32 fitting pixels; a tile is darkening when its own colour C moves its
  // darkened level by at most 4 t. Stellar Blade's band (033): 98.9% of
  // tiles within 4 t; the Witcher 3 wheel pair 99.5%; Hogwarts Legacy's
  // tinted minigame panel only 14.2% (median 12 t).
  inline constexpr std::uint32_t tile = contract::tile, tile_min_pixels = contract::tile_min_pixels;
  inline constexpr float tile_effect_scale = 4.f;
  // A tile whose pooled pre-UI values have no spread (variance at most this
  // share of their squared mean, a relative deviation below 0.1%: a flat
  // grey or black tile) has no regression and no verdict. Relative, so that
  // dark PQ scenes (linear values near 1e-4) are judged like bright ones,
  // and far above float rounding when the GPU reduces centered sums.
  inline constexpr double tile_min_relative_variance = 1e-6;
  inline constexpr std::uint32_t min_neighbourhood = ui_detection::change_set::min_neighbourhood;
  // The contract's integer mirrors, which the shader reads.
  static_assert(opacity_tolerance == static_cast<float>(contract::opacity_tolerance_255) / 255.f &&
    change_set_tolerance_scale == static_cast<float>(contract::change_set_scale) &&
    change_set_tolerance_scale == static_cast<float>(ui_detection::change_set::inferred_scale / 2u) &&
    step == 1.f / static_cast<float>(contract::step_inverse) &&
    tile_effect_scale == static_cast<float>(contract::tile_effect_scale) &&
    tile_min_relative_variance == static_cast<double>(contract::min_variance_ppm) * 1e-6);

  // Colour spaces of BUFFER_COLOR_SPACE.
  namespace color_space {
    inline constexpr std::uint32_t srgb = 1u, scrgb = 2u, pq = 3u;
  }

  // The sources the rule acts on (the contract's): the UI color tag (2) and
  // the offscreen UI layer (10) by opacity, the HUD-less difference (5) and
  // the pre-UI change set (12) by change set. Full-frame sources (6, 8, 11)
  // and alpha-only ones (1, 3, 4) are never touched.
  using contract::change_set_source;
  using contract::eligible;
  using contract::opacity_source;
  using contract::tiles;

  // The fit space: PQ (3) the channel-wise SMPTE ST 2084 EOTF without a
  // primaries matrix (1 = 10000 nits; the shader's SunshineDecodePQ applies
  // the matrix and x125 and is not this), identity otherwise. PQ blends in
  // linear light (Witcher 3: 83% of the wheel pair's pixels fit a darkening
  // within 1 t in linear, 4.5% in PQ code); sRGB UNORM blends are a scale in
  // either space, and scRGB is linear.
  namespace pq {
    inline constexpr float m1 = 2610.f / 16384.f, m2 = 2523.f / 4096.f * 128.f, c1 = 3424.f / 4096.f,
      c2 = 2413.f / 4096.f * 32.f, c3 = 2392.f / 4096.f * 32.f;
    inline float eotf(float e) {
      const float p = std::pow(std::clamp(e, 0.f, 1.f), 1.f / m2);
      return std::pow(std::max(p - c1, 0.f) / (c2 - c3 * p), 1.f / m1);
    }
    inline float inverse_eotf(float y) {
      const float p = std::pow(std::clamp(y, 0.f, 1.f), m1);
      return std::pow((c1 + c2 * p) / (1.f + c3 * p), m2);
    }
  }
  inline float fit(float encoded, std::uint32_t space) { return space == color_space::pq ? pq::eotf(encoded) : encoded; }
  inline float unfit(float value, std::uint32_t space) {
    return space == color_space::pq ? pq::inverse_eotf(value) : value;
  }

  using rgb = std::array<float, 3>;
  inline float peak(const rgb &c) { return std::max({std::abs(c[0]), std::abs(c[1]), std::abs(c[2])}); }
  inline bool finite(const rgb &c) { return std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2]); }
  // The shader's SunshineColorDifference(final, pre): the largest channel
  // difference, relative to the final colour's peak above 1 in scRGB.
  inline float color_difference(const rgb &final, const rgb &pre, std::uint32_t space) {
    const float scale = space == color_space::scrgb ? std::max(1.f, peak(final)) : 1.f;
    return std::max({std::abs(final[0] - pre[0]), std::abs(final[1] - pre[1]), std::abs(final[2] - pre[2])}) / scale;
  }

  // One pixel's evidence: ui in the decided source's mask; colour_raw raw
  // colour UI (what a neighbour touching it reads); dark a darkening pixel of
  // the mask; fits (change sets) a pixel of the mask whose darkening fit is
  // within the residual bound, darker or not, which the tile regression
  // pools; a the opacity and sigma its uncertainty, which every pixel
  // carries for its neighbours' step test unless it is not finite (then it
  // is no UI and takes part in no step).
  struct evidence {
    bool finite{}, ui{}, colour_raw{}, dark{}, fits{};
    float a{}, sigma{};
  };

  // Opacity evidence: the source's colour (premultiplied for the layer) and
  // alpha. UI where alpha is positive; colour when any channel's magnitude
  // exceeds the tolerance (the absolute value, so that a negative scRGB
  // channel counts as colour): opacity_tolerance_for(linear), linear for a
  // float source.
  inline evidence opacity(const rgb &colour, float alpha, bool linear = false) {
    if (!finite(colour) || !std::isfinite(alpha)) return {};
    evidence e;
    e.finite = true;
    e.ui = alpha > 0.f;
    e.colour_raw = e.ui && peak(colour) > opacity_tolerance_for(linear);
    e.dark = e.fits = e.ui && !e.colour_raw;
    e.a = alpha;
    return e;
  }

  // The darkening fit of a change-set pixel: s_raw = sum f(P) f(L) / sum
  // f(L)^2 (zero for a black L), s = clamp(s_raw, 0, 1), and the residual
  // R = max_c |P_c - f^-1(s f(L_c))| in the pair's encoding (scRGB: relative
  // to the presented peak above 1, as SunshineColorDifference).
  struct darkening_fit {
    float s_raw{}, s{}, residual{};
  };
  inline darkening_fit fit_darkening(const rgb &presented, const rgb &pre_ui, std::uint32_t space) {
    darkening_fit r;
    float num = 0.f, den = 0.f;
    rgb fl{};
    for (std::size_t c = 0; c != 3; ++c) {
      fl[c] = fit(pre_ui[c], space);
      num += fit(presented[c], space) * fl[c];
      den += fl[c] * fl[c];
    }
    r.s_raw = den > 0.f ? num / den : 0.f;
    r.s = std::clamp(r.s_raw, 0.f, 1.f);
    for (std::size_t c = 0; c != 3; ++c)
      r.residual = std::max(r.residual, std::abs(presented[c] - unfit(r.s * fl[c], space)));
    if (space == color_space::scrgb) r.residual /= std::max(1.f, peak(presented));
    return r;
  }
  // Whether a fit's residual is within change_set_tolerance_scale t, and
  // whether it is a pure darkening at threshold t: such a fit and not
  // brighter than the pre-UI pixel.
  inline bool fits(const darkening_fit &f, float t) { return f.residual <= change_set_tolerance_scale * t; }
  inline bool darkening(const darkening_fit &f, float t) { return f.s_raw < 1.f && fits(f, t); }

  // The opacity uncertainty of a = 1 - s from one threshold step t at the
  // pre-UI pixel's peak e: PQ min(1, (f(e + t) - f(e)) / max(f(e), 1e-12)),
  // the identity encodings t / max(e, t).
  inline float sigma(const rgb &pre_ui, float t, std::uint32_t space) {
    const float e = peak(pre_ui);
    if (space == color_space::pq) {
      const float lo = pq::eotf(e), hi = pq::eotf(std::min(e + t, 1.f));
      return std::min(1.f, (hi - lo) / std::max(lo, 1e-12f));
    }
    return t / std::max(e, t);
  }

  // Change-set evidence: the presented pixel P, its pre-UI pixel L, the pair
  // threshold t, the scale k (1 for a declared pair, inferred_scale for the
  // pre-UI change set) and whether the pixel is in the source's mask (source
  // 5: changed beyond t; source 12: changed beyond k t with min_neighbourhood
  // changed pixels in its 3x3 window). Raw colour is changed beyond k t
  // without the 3x3 rule and not a darkening.
  inline evidence change_set(const rgb &presented, const rgb &pre_ui, float t, float k, bool in_mask, std::uint32_t space) {
    if (!finite(presented) || !finite(pre_ui)) return {};
    const auto f = fit_darkening(presented, pre_ui, space);
    const bool dims = darkening(f, t);
    evidence e;
    e.finite = true;
    e.ui = in_mask;
    e.colour_raw = color_difference(presented, pre_ui, space) > k * t && !dims;
    e.dark = in_mask && dims;
    e.fits = in_mask && fits(f, t);
    e.a = 1.f - f.s;
    e.sigma = sigma(pre_ui, t, space);
    return e;
  }

  // Whether two neighbours differ by a sharp opacity step.
  inline bool step_between(const evidence &x, const evidence &n) {
    return x.finite && n.finite && std::abs(x.a - n.a) - x.sigma - n.sigma >= step;
  }

  // The tile regression's sums over a tile's fitting pixels (evidence::fits),
  // pooled channels: x = f(L_c), y = f(P_c), three samples per pixel.
  struct tile_sums {
    std::uint32_t pixels{};
    double sx{}, sy{}, sxx{}, sxy{};
    void add(const rgb &presented, const rgb &pre_ui, std::uint32_t space) {
      ++pixels;
      for (std::size_t c = 0; c != 3; ++c) {
        const double x = fit(pre_ui[c], space), y = fit(presented[c], space);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
      }
    }
  };
  enum class verdict : std::uint8_t { none, darkening, colour };
  // A tile's effect E: m = cov / var, C = mean(y) - m mean(x), base =
  // clamp(m mean(x), 0, 1), E = |f^-1(clamp(base + C, 0, 1)) - f^-1(base)| / t
  // (scRGB: also over max(1, base)); negative without a verdict (fewer than
  // tile_min_pixels fitting pixels, or no spread).
  inline double tile_effect(const tile_sums &sums, float t, std::uint32_t space) {
    if (sums.pixels < tile_min_pixels || !(t > 0.f)) return -1.;
    const double n = 3. * sums.pixels, mx = sums.sx / n, my = sums.sy / n;
    const double var = sums.sxx / n - mx * mx, cov = sums.sxy / n - mx * my;
    if (!(var > tile_min_relative_variance * mx * mx) || !std::isfinite(var) || !std::isfinite(cov)) return -1.;
    const double m = cov / var, offset = my - m * mx, base = std::clamp(m * mx, 0., 1.);
    const auto encode = [&](double value) {
      return space == color_space::pq ? static_cast<double>(pq::inverse_eotf(static_cast<float>(value))) : value;
    };
    double effect = std::abs(encode(std::clamp(base + offset, 0., 1.)) - encode(base)) / t;
    if (space == color_space::scrgb) effect /= std::max(1., base);
    return std::isfinite(effect) ? effect : -1.;
  }
  inline verdict tile_verdict(const tile_sums &sums, float t, std::uint32_t space) {
    const double effect = tile_effect(sums, t, space);
    return effect < 0. ? verdict::none : effect <= tile_effect_scale ? verdict::darkening : verdict::colour;
  }

  // The per-pixel classes of the reference.
  enum class pixel_class : std::uint8_t { not_ui, colour, kept, unpinned };
  struct counts {
    std::uint64_t covered{}, darkening{}, unpinned{}, kept{}, tiles_darkening{}, tiles_colour{};
    // Change sets: the darkening regions, and those that unpin.
    std::uint64_t regions{}, regions_unpinned{};
  };
  struct result {
    std::uint32_t width{}, height{};
    std::vector<pixel_class> classes;
    // Per pixel: on a step or touching raw colour (STRUCT), and kept by the
    // 3x3 rule; per tile (row-major, tiles_x by tiles_y; change sets only,
    // opacity sources leave them empty): the verdict, a colour verdict in its
    // 3x3 ring, a darkening pixel in it, its region's index (-1 without a
    // darkening pixel) and whether its region unpins.
    std::vector<std::uint8_t> structure, kept;
    std::uint32_t tiles_x{}, tiles_y{};
    std::vector<verdict> verdicts;
    std::vector<std::uint8_t> ring, candidate, region;
    std::vector<std::int32_t> component;
    // Opacity sources: no pixel of the source carried colour, so nothing
    // unpins (word dk_kept's darkening::colourless).
    bool colourless{};
    counts n;
  };

  // The reference over a whole image's evidence (row-major). With a tile
  // gate (change sets: presented and pre-UI colours, threshold and space),
  // a darkening pixel unpins only in a region that unpins; without one (an
  // opacity source), only when some pixel of the source carries colour.
  struct tile_gate {
    const std::vector<rgb> *presented{}, *pre_ui{};
    float t{};
    std::uint32_t space{};
  };
  inline result classify(std::uint32_t width, std::uint32_t height, const std::vector<evidence> &e, const tile_gate *gate) {
    result r;
    r.width = width;
    r.height = height;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    r.classes.assign(count, pixel_class::not_ui);
    r.structure.assign(count, 0u);
    r.kept.assign(count, 0u);
    const auto at = [&](std::uint32_t x, std::uint32_t y) { return static_cast<std::size_t>(y) * width + x; };
    constexpr std::array<std::array<int, 2>, 4> four{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
    for (std::uint32_t y = 0; y != height; ++y) {
      for (std::uint32_t x = 0; x != width; ++x) {
        const auto &p = e[at(x, y)];
        if (!p.dark) continue;
        bool structured = false;
        for (const auto &d : four) {
          const int nx = static_cast<int>(x) + d[0], ny = static_cast<int>(y) + d[1];
          if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
          const auto &n = e[at(static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny))];
          if (n.colour_raw || step_between(p, n)) {
            structured = true;
            break;
          }
        }
        r.structure[at(x, y)] = structured ? 1u : 0u;
      }
    }
    for (std::uint32_t y = 0; y != height; ++y) {
      for (std::uint32_t x = 0; x != width; ++x) {
        if (!r.structure[at(x, y)]) continue;
        std::uint32_t around = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = static_cast<int>(x) + dx, ny = static_cast<int>(y) + dy;
            if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
            around += r.structure[at(static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny))];
          }
        }
        r.kept[at(x, y)] = around >= min_neighbourhood ? 1u : 0u;
      }
    }
    if (gate) {
      r.tiles_x = tiles(width);
      r.tiles_y = tiles(height);
      const std::size_t tile_count = static_cast<std::size_t>(r.tiles_x) * r.tiles_y;
      const auto tile_of = [&](std::uint32_t x, std::uint32_t y) {
        return static_cast<std::size_t>(y / tile) * r.tiles_x + x / tile;
      };
      std::vector<tile_sums> sums(tile_count);
      r.candidate.assign(tile_count, 0u);
      for (std::uint32_t y = 0; y != height; ++y) {
        for (std::uint32_t x = 0; x != width; ++x) {
          // Kept darkening pixels (edges, outlines, the rims of colour UI)
          // stay pinned anyway and join no region; they still count in the
          // regression, where a tint over a textured scene (kept, as its
          // opacity steps with the scene) shows its colour.
          const auto &p = e[at(x, y)];
          if (p.fits) sums[tile_of(x, y)].add((*gate->presented)[at(x, y)], (*gate->pre_ui)[at(x, y)], gate->space);
          if (p.dark && !r.kept[at(x, y)]) r.candidate[tile_of(x, y)] = 1u;
        }
      }
      r.verdicts.resize(tile_count);
      for (std::size_t i = 0; i != tile_count; ++i) {
        r.verdicts[i] = tile_verdict(sums[i], gate->t, gate->space);
        if (r.verdicts[i] == verdict::darkening) ++r.n.tiles_darkening;
        else if (r.verdicts[i] == verdict::colour) ++r.n.tiles_colour;
      }
      const auto neighbours = [&](std::uint32_t tx, std::uint32_t ty, auto &&visit) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = static_cast<int>(tx) + dx, ny = static_cast<int>(ty) + dy;
            if (nx < 0 || ny < 0 || nx >= static_cast<int>(r.tiles_x) || ny >= static_cast<int>(r.tiles_y)) continue;
            visit(static_cast<std::size_t>(ny) * r.tiles_x + static_cast<std::size_t>(nx));
          }
        }
      };
      r.ring.assign(tile_count, 0u);
      for (std::size_t i = 0; i != tile_count; ++i) {
        if (r.verdicts[i] == verdict::colour)
          neighbours(static_cast<std::uint32_t>(i % r.tiles_x), static_cast<std::uint32_t>(i / r.tiles_x),
            [&](std::size_t j) { r.ring[j] = 1u; });
      }
      // The darkening regions: 8-connected tiles holding darkening pixels. A
      // region unpins with a proven tile and no colour-ringed tile, and only
      // when the shader's region pass can hold the tile grid.
      r.component.assign(tile_count, -1);
      r.region.assign(tile_count, 0u);
      const bool supported = ui_detection::darkening::region_supported(width, height);
      std::vector<std::size_t> members;
      for (std::size_t seed = 0; seed != tile_count; ++seed) {
        if (!r.candidate[seed] || r.component[seed] >= 0) continue;
        const auto index = static_cast<std::int32_t>(r.n.regions++);
        members.assign(1, seed);
        r.component[seed] = index;
        bool proof = false, poison = false;
        for (std::size_t next = 0; next != members.size(); ++next) {
          const std::size_t i = members[next];
          poison = poison || r.ring[i] != 0u;
          proof = proof || (r.verdicts[i] == verdict::darkening && !r.ring[i]);
          neighbours(static_cast<std::uint32_t>(i % r.tiles_x), static_cast<std::uint32_t>(i / r.tiles_x), [&](std::size_t j) {
            if (!r.candidate[j] || r.component[j] >= 0) return;
            r.component[j] = index;
            members.push_back(j);
          });
        }
        if (!supported || !proof || poison) continue;
        ++r.n.regions_unpinned;
        for (const std::size_t i : members) r.region[i] = 1u;
      }
    } else {
      r.colourless = std::none_of(e.begin(), e.end(), [](const evidence &p) { return p.colour_raw; });
    }
    for (std::uint32_t y = 0; y != height; ++y) {
      for (std::uint32_t x = 0; x != width; ++x) {
        const auto i = at(x, y);
        const auto &p = e[i];
        if (!p.ui) continue;
        ++r.n.covered;
        if (!p.dark) {
          r.classes[i] = pixel_class::colour;
          continue;
        }
        ++r.n.darkening;
        const bool proven = gate ? r.region[static_cast<std::size_t>(y / tile) * r.tiles_x + x / tile] != 0u : !r.colourless;
        if (!r.kept[i] && proven) {
          r.classes[i] = pixel_class::unpinned;
          ++r.n.unpinned;
        } else {
          r.classes[i] = pixel_class::kept;
          ++r.n.kept;
        }
      }
    }
    return r;
  }

  // The reference of an opacity source over a whole image: the source's
  // colour (premultiplied for the layer) and alpha per pixel, and whether
  // its format is float (linear colour).
  inline result reference_opacity(std::uint32_t width, std::uint32_t height, const std::vector<rgb> &colour,
      const std::vector<float> &alpha, bool linear = false) {
    std::vector<evidence> e(colour.size());
    for (std::size_t i = 0; i != e.size(); ++i) e[i] = opacity(colour[i], alpha[i], linear);
    return classify(width, height, e, nullptr);
  }

  // The mask of a change-set source: changed beyond k t (finite pixels),
  // and, with a neighbourhood rule (the pre-UI change set), at least
  // min_neighbourhood changed pixels in the 3x3 window, itself included
  // (out-of-frame neighbours unchanged).
  inline std::vector<std::uint8_t> change_set_mask(std::uint32_t width, std::uint32_t height,
      const std::vector<rgb> &presented, const std::vector<rgb> &pre_ui, float t, float k, std::uint32_t space,
      bool neighbourhood) {
    const std::size_t count = static_cast<std::size_t>(width) * height;
    std::vector<std::uint8_t> changed(count, 0u);
    for (std::size_t i = 0; i != count; ++i)
      changed[i] = finite(presented[i]) && finite(pre_ui[i]) && color_difference(presented[i], pre_ui[i], space) > k * t;
    if (!neighbourhood) return changed;
    std::vector<std::uint8_t> mask(count, 0u);
    for (std::uint32_t y = 0; y != height; ++y) {
      for (std::uint32_t x = 0; x != width; ++x) {
        if (!changed[static_cast<std::size_t>(y) * width + x]) continue;
        std::uint32_t around = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = static_cast<int>(x) + dx, ny = static_cast<int>(y) + dy;
            if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
            around += changed[static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx)];
          }
        }
        mask[static_cast<std::size_t>(y) * width + x] = around >= min_neighbourhood ? 1u : 0u;
      }
    }
    return mask;
  }

  // The reference of a change-set source over a whole image: presented and
  // pre-UI colours, the source's mask, the pair threshold t, the scale k and
  // the colour space.
  inline result reference_change_set(std::uint32_t width, std::uint32_t height, const std::vector<rgb> &presented,
      const std::vector<rgb> &pre_ui, const std::vector<std::uint8_t> &mask, float t, float k, std::uint32_t space) {
    std::vector<evidence> e(presented.size());
    for (std::size_t i = 0; i != e.size(); ++i) e[i] = change_set(presented[i], pre_ui[i], t, k, mask[i] != 0u, space);
    const tile_gate gate{&presented, &pre_ui, t, space};
    return classify(width, height, e, &gate);
  }

  // One committed sample's darkening (decision texel 15 .z/.w): the decided
  // source, the session's switch (UIPinOnlyUI), the source's covered pixels
  // (texel 0), its unpinned and kept darkening pixels, and whether an
  // opacity source carried no colour (darkening::colourless in dk_kept).
  struct sample {
    std::uint32_t source{};
    bool enabled{};
    std::uint32_t covered{}, unpinned{}, kept{};
    bool colourless{};
  };
  // A sample from decision words dk_unpinned and dk_kept.
  inline sample sample_of(std::uint32_t source, bool enabled, std::uint32_t covered, std::uint32_t unpinned,
      std::uint32_t kept_word) {
    return {source, enabled, covered, unpinned, kept_word & ~contract::colourless, (kept_word & contract::colourless) != 0u};
  }
  // The log text of a sample (game3d_log_report.py parses it).
  inline std::string log_text(const sample &s) {
    std::string text = "Sunshine UI darkening: source=";
    text.append(std::to_string(s.source)).append(" UIPinOnlyUI=").append(s.enabled ? "1" : "0");
    text.append(" covered=").append(std::to_string(s.covered));
    text.append(" unpinned=").append(std::to_string(s.unpinned));
    text.append(" kept=").append(std::to_string(s.kept));
    text.append(" colourless=").append(s.colourless ? "1" : "0");
    return text;
  }
  // The line is throttled as fix 3's change-set line: logged when the
  // source, the switch, whether anything unpins or the colourless bit
  // changes, else at most once per log_interval_ms; the counter group
  // darkening counts every sample.
  inline constexpr std::uint64_t log_interval_ms = 1000;
  struct log_state {
    bool logged{};
    std::uint64_t last_ms{};
    std::uint32_t source{};
    bool enabled{}, unpins{}, colourless{};
  };
  inline bool log_due(log_state &state, const sample &s, std::uint64_t now_ms) {
    const bool changed = !state.logged || state.source != s.source || state.enabled != s.enabled ||
      state.unpins != (s.unpinned > 0u) || state.colourless != s.colourless;
    if (!changed && now_ms >= state.last_ms && now_ms - state.last_ms < log_interval_ms) return false;
    state = {true, now_ms, s.source, s.enabled, s.unpinned > 0u, s.colourless};
    return true;
  }
}
