// SPDX-License-Identifier: GPL-3.0-only
// Stage A1 of world-anchored UI planes (docs/game3d-anchored-ui-planes.md):
// the single-frame pieces that need no dump or GPU. The anchored pin's
// envelope with every target at the sentinel equals the UI pin band
// (test_game3d_ui_pin_band.h) within 1e-5 px; the feasibility resolve of
// game3d_ui_anchor.h never leaves L > U on randomized elements, while the
// unresolved targets do; D(g) is exact; the cell grid stays within its cap at
// every extent; and labelled 21:9 and 32:9 frames give one element per
// labelled element.
#include "game3d_ui_anchor.h"
#include "test_game3d_ui_pin_band.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  namespace anchor = sunshine_game3d::ui_anchor;
  using sunshine_game3d_test::ui_pin_band_detail::band;
  using sunshine_game3d_test::ui_pin_band_detail::envelope;

  constexpr double gain = 8;  // SUNSHINE_UI_SOFT_PIN_GAIN

  void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
  }

  // splitmix64: identical on every standard library, unlike <random>'s distributions.
  struct rng {
    std::uint64_t state;
    std::uint64_t next() {
      std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
      return z ^ (z >> 31);
    }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }
    double between(double low, double high) { return low + (high - low) * uniform(); }
    unsigned below(unsigned n) { return unsigned(next() % n); }
  };

  struct extent {
    unsigned width, height;
    const char *name;
  };
  // 16:9, 21:9 and 32:9 fixtures for the envelope and the resolve.
  constexpr extent fixtures[] {{256, 144, "16:9"}, {336, 144, "21:9"}, {512, 144, "32:9"}};

  // A field the horizontal limiter could produce: 0.5/W-Lipschitz along rows.
  std::vector<float> lipschitz_field(rng &r, unsigned width, unsigned height, double bound) {
    std::vector<double> v(size_t(width) * height);
    for (auto &value : v) value = r.between(-bound, bound);
    const double step = .5 / width;
    for (unsigned y = 0; y < height; ++y) {
      double *row = v.data() + size_t(y) * width;
      for (unsigned x = 1; x < width; ++x) row[x] = std::max(row[x], row[x - 1] - step);
      for (unsigned x = width - 1; x-- > 0;) row[x] = std::max(row[x], row[x + 1] - step);
    }
    // Rounding to float32 may break the bound by an ulp; the slope checks allow it.
    return std::vector<float>(v.begin(), v.end());
  }

  double steepest(const std::vector<double> &field, unsigned width, unsigned height) {
    double result = 0;
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x + 1 < width; ++x) result = std::max(result, std::abs(field[size_t(y) * width + x + 1] - field[size_t(y) * width + x]));
    return result;
  }

  // ---------------------------------------------------------------- cell grid

  void cell_grid_stays_within_its_cap() {
    require(anchor::cell_side(3840, 2160) == 35 && anchor::cell_side(2560, 1440) == 23, "cell sides differ from 35 px at 4K and 23 px at 1440p");
    const auto cells = [](unsigned w, unsigned h, unsigned s) { return std::uint64_t((w + s - 1) / s) * ((h + s - 1) / s); };
    unsigned checked = 0;
    for (unsigned w = 64; w <= 3840; w += 64) {
      for (unsigned h = 64; h <= 3840; h += 64) {
        const unsigned s = anchor::cell_side(w, h);
        require(cells(w, h, s) <= anchor::max_cells, "a cell grid exceeds 7168 cells at " + std::to_string(w) + "x" + std::to_string(h));
        require(s == 1 || cells(w, h, s - 1) > anchor::max_cells, "a cell side is not the smallest at " + std::to_string(w) + "x" + std::to_string(h));
        ++checked;
      }
    }
    const extent wide[] {{3440, 1440, "21:9"}, {2560, 1080, "21:9"}, {3840, 1080, "32:9"}, {3840, 3840, "1:1 bound"}};
    for (const auto &e : wide) {
      const unsigned s = anchor::cell_side(e.width, e.height);
      std::printf("  %s %ux%u: %u px cells, %llu cells\n", e.name, e.width, e.height, s,
        static_cast<unsigned long long>(cells(e.width, e.height, s)));
    }
    std::printf("  %u extents up to 3840x3840 within 7168 cells, each side the smallest\n", checked);
  }

  // ---------------------------------------------------------------- envelope against band

  std::vector<float> soft_mask(rng &r, unsigned width, unsigned height) {
    std::vector<float> mask(size_t(width) * height, 0.f);
    for (unsigned y = height / 5; y < height * 4 / 5; ++y)
      for (unsigned x = width / 6; x < width * 5 / 6; ++x) {
        const unsigned pick = r.below(10);
        mask[size_t(y) * width + x] = pick < 4 ? 0.f : pick < 7 ? float(r.below(255) + 1) / float(255 * gain) : float(r.below(1024) + 1) / 1024.f;
      }
    // Non-finite alpha never pins.
    mask[size_t(height / 2) * width + width / 2] = std::nanf("");
    return mask;
  }

  void all_sentinel_envelope_equals_band() {
    rng r {0x51a7e11u};
    for (const auto &e : fixtures) {
      const unsigned w = e.width, h = e.height;
      const double bound = .02, plane = .006;
      const auto field = lipschitz_field(r, w, h, bound);
      const auto mask = soft_mask(r, w, h);
      const auto reference = band(field, mask, w, h, plane, gain);
      const std::vector<double> sentinels(field.size(), anchor::sentinel), planes(field.size(), plane);
      double worst = 0;
      for (const auto *targets : {static_cast<const std::vector<double> *>(nullptr), &sentinels, &planes}) {
        const auto out = envelope(field, mask, targets, w, h, plane, gain);
        for (size_t i = 0; i < field.size(); ++i) {
          worst = std::max(worst, std::abs(out.field[i] - reference[i]) * w);
          require(!(out.lower[i] > out.upper[i]), std::string(e.name) + " all-sentinel envelope has L > U");
        }
      }
      require(worst <= 1e-5, std::string(e.name) + " all-sentinel envelope differs from band() by " + std::to_string(worst) + " px");
      std::printf("  %s %ux%u: all-sentinel envelope within %.3g px of band()\n", e.name, w, h, worst);
    }
  }

  // ---------------------------------------------------------------- D(g)

  // Two rigid single-texel targets g empty columns apart: L <= U exactly up to
  // D(g), and L > U just beyond it.
  void proximity_is_exact() {
    const unsigned w = 64, h = 1;
    const double plane = 0;
    for (int gap = 0; gap <= 6; ++gap) {
      const double d = anchor::proximity_uv(gap, w);
      for (const double extra : {0., 1e-3 / w}) {
        std::vector<float> field(w, 0.f), mask(w, 0.f);
        std::vector<double> targets(w, anchor::sentinel);
        const unsigned a = 20, b = a + unsigned(gap) + 1;
        mask[a] = mask[b] = 1.f;
        targets[a] = 0;
        targets[b] = d + extra;
        const auto out = envelope(field, mask, &targets, w, h, plane, gain);
        double excess = -1;
        for (unsigned x = 0; x < w; ++x) excess = std::max(excess, out.lower[x] - out.upper[x]);
        if (extra == 0) require(excess <= 1e-15, "D(" + std::to_string(gap) + ") leaves L > U");
        else require(excess > 0, "D(" + std::to_string(gap) + ") is not the largest feasible difference");
      }
    }
    std::printf("  D(g) = 0.5*max(g-1,0)/W is the largest feasible difference for g = 0..6\n");
  }

  // ---------------------------------------------------------------- randomized resolves

  struct bar {
    int x0, y0, x1, y1;  // core texels, inclusive
    bool halo;
  };

  std::vector<float> rasterize(const std::vector<bar> &bars, unsigned width, unsigned height, rng &r) {
    std::vector<float> mask(size_t(width) * height, 0.f);
    for (const auto &b : bars) {
      for (int y = b.y0 - 1; y <= b.y1 + 1; ++y) {
        for (int x = b.x0 - 1; x <= b.x1 + 1; ++x) {
          if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) continue;
          const bool core = x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
          if (!core && !b.halo) continue;
          auto &m = mask[size_t(y) * width + size_t(x)];
          const float value = core ? 1.f : float(r.between(.01, 1.));
          m = std::max(m, value);
        }
      }
    }
    return mask;
  }

  // HUD | X | A touching in one row, A nearest: X has the plane on one side and
  // A on the other, so it is demoted, and the repeat pulls A to the plane too.
  void demotion_repeats_the_resolve() {
    const unsigned w = 64;
    const double plane = .001, bound = .01;
    std::vector<anchor::resolve_item> items(3);
    items[0].global = true;
    items[0].outlines = {{10, 0, 19, 0}};
    items[1].desired = .004;
    items[1].order = .004;
    items[1].outlines = {{20, 0, 29, 0}};
    items[2].desired = .008;
    items[2].order = .008;
    items[2].lower = .007;
    items[2].outlines = {{30, 0, 39, 0}};
    const auto out = anchor::resolve(items, plane, bound, w);
    require(out.demoted[1] && !out.demoted[2] && out.floor_dropped[2] && out.rounds == 2, "a squeezed element was not demoted once");
    require(out.value[1] == plane && out.value[2] == plane, "the demotion's repeat did not pull its neighbour to the plane");
  }

  void randomized_resolves_keep_lower_below_upper() {
    demotion_repeats_the_resolve();
    rng r {0x7e501fe5u};
    unsigned trials = 0, unresolved_violations = 0, demotions = 0, drops = 0, constrained = 0;
    double worst_excess_px = -1e9, worst_slope = 0;
    for (const auto &e : fixtures) {
      const unsigned w = e.width, h = e.height;
      for (unsigned trial = 0; trial < 150; ++trial, ++trials) {
        const double bound = r.between(.002, .04), plane = r.between(-.5, .5) * bound;
        // Bars in a few shared row bands, so neighbours often share rows at small gaps.
        std::vector<bar> bars;
        const unsigned count = 2 + r.below(14);
        for (unsigned i = 0; i < count; ++i) {
          const int y0 = int(8 + 12 * r.below(10) + r.below(4)), x0 = int(r.below(w - 40));
          bars.push_back({x0, y0, x0 + 3 + int(r.below(30)), y0 + int(r.below(5)), r.below(2) == 0});
        }
        const auto mask = rasterize(bars, w, h, r);
        const auto field = lipschitz_field(r, w, h, bound);
        const auto grid = anchor::cells_of(mask.data(), w, h);
        const auto elements = anchor::elements_of(grid);
        std::vector<anchor::resolve_item> items(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) {
          auto &item = items[i];
          for (const auto c : elements[i].cells) item.outlines.push_back(grid.cells[c].outline);
          item.global = r.below(10) < 3;
          item.desired = r.between(-bound, bound);
          item.order = r.between(-bound, bound);
          if (r.below(10) < 4) item.lower = r.between(-bound, 1.2 * bound);
        }
        const auto resolved = anchor::resolve(items, plane, bound, w);
        // The resolve's own invariants.
        for (size_t i = 0; i < items.size(); ++i) {
          const double v = resolved.value[i];
          demotions += resolved.demoted[i];
          drops += resolved.floor_dropped[i];
          if (items[i].global || resolved.demoted[i]) require(v == plane, "a global or demoted element left the plane");
          require(v >= -bound && v <= bound, "a resolved target left [-C, C]");
          if (!items[i].global && !resolved.demoted[i] && !anchor::is_sentinel(items[i].lower) && !resolved.floor_dropped[i])
            require(v >= items[i].lower, "a resolved target dropped below its floor without dropping it");
          for (size_t j = 0; j < items.size(); ++j) {
            const double d = anchor::proximity_between(items[i], items[j], w);
            if (i == j || !std::isfinite(d)) continue;
            ++constrained;
            require(std::abs(v - resolved.value[j]) <= d + 1e-15, "two resolved neighbours are farther apart than D(g)");
          }
        }
        // Per-texel targets: the resolved values, and as the control the raw desired values.
        for (const bool resolved_targets : {true, false}) {
          std::vector<double> targets(mask.size(), anchor::sentinel);
          for (size_t i = 0; i < elements.size(); ++i) {
            const bool global = items[i].global || (resolved_targets && resolved.demoted[i]);
            const double t = global ? anchor::sentinel : resolved_targets ? resolved.value[i] : items[i].desired;
            for (const auto c : elements[i].cells) {
              const auto &o = grid.cells[c].outline;
              for (int y = o.y0; y <= o.y1; ++y)
                for (int x = o.x0; x <= o.x1; ++x)
                  if (mask[size_t(y) * w + size_t(x)] > 0.f) targets[size_t(y) * w + size_t(x)] = t;
            }
          }
          const auto out = envelope(field, mask, &targets, w, h, plane, gain);
          double excess = -1e9;
          for (size_t i = 0; i < mask.size(); ++i) excess = std::max(excess, (out.lower[i] - out.upper[i]) * w);
          if (resolved_targets) {
            worst_excess_px = std::max(worst_excess_px, excess);
            require(excess <= 1e-9, std::string(e.name) + " trial " + std::to_string(trial) + ": a resolved envelope has L > U by " +
              std::to_string(excess) + " px");
          } else {
            unresolved_violations += excess > 1e-9;
          }
          // Either way the field is a min/max of 0.5/W-Lipschitz functions.
          const double slope = steepest(out.field, w, h) * w;
          worst_slope = std::max(worst_slope, slope);
          require(slope <= .5 * (1 + 1e-4), std::string(e.name) + " an envelope exceeds the 0.5/W slope: " + std::to_string(slope) + " px");
        }
      }
    }
    require(unresolved_violations > 0, "no unresolved control trial had L > U: the randomized fixture does not test feasibility");
    require(constrained > 0 && demotions > 0 && drops > 0, "the randomized fixture never constrained, demoted or dropped a floor");
    std::printf("  %u trials (16:9, 21:9, 32:9): resolved L - U at most %.3g px; unresolved targets gave L > U in %u trials; "
      "%u neighbour pairs, %u floors dropped, %u demotions; slope at most %.9g px\n", trials, worst_excess_px, unresolved_violations,
      constrained, drops, demotions, worst_slope);
  }

  // ---------------------------------------------------------------- labelled frames

  // A full-resolution frame at 21:9 and 32:9 with labelled elements: each label
  // must be exactly one element, with the expected candidate status.
  void labelled_frames_give_one_element_per_label() {
    struct label {
      const char *name;
      std::vector<bar> pieces;  // glyphs or segments of one element
      const char *refusal;      // expected candidate refusal, or null
    };
    for (const extent e : {extent {3440, 1440, "21:9"}, extent {3840, 1080, "32:9"}}) {
      const int w = int(e.width), h = int(e.height), s = int(anchor::cell_side(e.width, e.height));
      const int cx = w / 2, cy = h / 2;
      std::vector<label> labels {
        // An enemy bar in the centre: one candidate.
        {"bar", {{cx - 300, cy - 200, cx - 150, cy - 190, true}}, nullptr},
        // A nameplate of glyphs closer than a cell: one element.
        {"nameplate", {{cx + 200, cy - 220, cx + 214, cy - 200, false}, {cx + 220, cy - 220, cx + 232, cy - 200, false},
                       {cx + 238, cy - 220, cx + 260, cy - 200, false}}, nullptr},
        // A crosshair of five pieces within one cell (RE9): one element.
        {"crosshair", {{cx - 1, cy - 1, cx + 1, cy + 1, false}, {cx - 12, cy, cx - 6, cy, false}, {cx + 6, cy, cx + 12, cy, false},
                       {cx, cy - 12, cx, cy - 6, false}, {cx, cy + 6, cx, cy + 12, false}}, nullptr},
        // A compass strip at the top edge: outside the central 75%.
        {"compass", {{cx - 400, 20, cx + 400, 40, true}}, "edge"},
        // A subtitle line at the bottom.
        {"subtitle", {{cx - 500, h - 90, cx + 500, h - 60, false}}, "edge"},
        // A panel taller than H/4 in the centre.
        {"panel", {{cx + 500, cy - h / 6, cx + 560, cy + h / 6, false}}, "oversize"},
        // A soft glow without a core texel.
        {"glow", {{cx - 600, cy + 150, cx - 560, cy + 160, false}}, "no_core"},
      };
      rng r {0x1abe1u};
      std::vector<float> mask(size_t(w) * size_t(h), 0.f);
      for (const auto &l : labels) {
        std::vector<float> one = rasterize(l.pieces, e.width, e.height, r);
        if (std::string(l.name) == "glow")
          for (auto &v : one) v = v > 0.f ? .05f : 0.f;
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = std::max(mask[i], one[i]);
      }
      const auto grid = anchor::cells_of(mask.data(), e.width, e.height);
      const auto elements = anchor::elements_of(grid);
      for (const auto &l : labels) {
        anchor::box area;
        for (const auto &p : l.pieces) area.add(anchor::box {p.x0 - 1, p.y0 - 1, p.x1 + 1, p.y1 + 1});
        std::vector<const anchor::element *> hits;
        for (const auto &el : elements)
          if (anchor::overlaps(el.outline, area)) hits.push_back(&el);
        require(hits.size() == 1, std::string(e.name) + " label " + l.name + " is " + std::to_string(hits.size()) + " elements");
        const char *refusal = anchor::candidate_refusal(hits[0]->outline, hits[0]->core, e.width, e.height);
        require((refusal == nullptr) == (l.refusal == nullptr) && (!refusal || std::string(refusal) == l.refusal),
          std::string(e.name) + " label " + l.name + " has candidate refusal " + (refusal ? refusal : "none"));
      }
      require(elements.size() == labels.size(), std::string(e.name) + " frame has elements without a label");
      std::printf("  %s %dx%d (%d px cells): %zu labels, one element each, candidate status as labelled\n", e.name, w, h, s, labels.size());
    }
  }
}  // namespace

int main() {
  const std::pair<const char *, void (*)()> groups[] {
    {"A1 cell grid within 7168 cells", cell_grid_stays_within_its_cap},
    {"A1 all-sentinel envelope equals band()", all_sentinel_envelope_equals_band},
    {"A1 D(g) is exact", proximity_is_exact},
    {"A1 randomized resolves never give L > U", randomized_resolves_keep_lower_below_upper},
    {"A1 one element per labelled element (21:9, 32:9)", labelled_frames_give_one_element_per_label},
  };
  try {
    for (const auto &[name, group] : groups) {
      group();
      std::printf("PASS %s\n", name);
    }
    std::printf("PASS UI anchor envelope: %zu groups\n", std::size(groups));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL UI anchor envelope: %s\n", error.what());
    return 1;
  }
}
