// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// World-anchored UI planes (docs/game3d-anchored-ui-planes.md): the
// double-precision mirror of the anchor passes. Stage A1's oracle and stage
// A4's offline classifier and sequence ctest use it; no add-on source includes
// it, so the live renderer still pins every UI texel toward the one plane pUI.
//
// Units: the unpinned displayed field h, the plane pUI, the display bound C,
// goals, floors and applied values are signed source-U displacements per eye
// (one output pixel is 1/W, positive is nearer); motion is in output pixels,
// from real frame N-1 to real frame N; rho is inverse depth, 0 at the far
// plane. The render grid (depth and motion vectors) is registered to the
// output frame without crop or jitter; a render texel's output pixel is
// floor((i + 0.5) * W / W_mv).
//
// The sentinel target (NaN) means "the global plane": a texel or element
// whose target is the sentinel pins exactly as today.
#include "game3d_ui_adaptive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sunshine_game3d::ui_anchor {
  // Cost caps and tuned values (the design's constants table).
  inline constexpr std::uint32_t max_cells = 7168;
  inline constexpr std::uint32_t max_tracks = 64;  // also the candidate cap
  inline constexpr std::uint32_t reach_sample_cap = 1024;
  inline constexpr std::uint32_t check_sample_cap = 4096;
  inline constexpr std::uint32_t n_min = 4;  // distinct motion texels
  inline constexpr double check_pass_factor = 2;
  // The conflict probe's margin, 0.05 * C (UI protection under Setup), and the
  // coverage-share half of its rule for one element: at most 20% of the core in
  // front of the floor, so the floor is the 80th percentile.
  inline constexpr double probe_margin = .05;
  inline constexpr std::uint32_t floor_percent = 100 - ui_adaptive::entry_conflict_percent;
  inline constexpr unsigned pin_group_rows = 8;
  inline constexpr unsigned quarter = 4;  // the motion check's log-depth ring
  inline constexpr std::uint32_t delta_bins = 8;  // delta = 2/W in quarter-pixel bins
  inline constexpr double sentinel = std::numeric_limits<double>::quiet_NaN();

  [[nodiscard]] inline bool is_sentinel(double value) {
    return std::isnan(value);
  }

  // Inclusive texel bounds; empty when x1 < x0 or y1 < y0.
  struct box {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;

    [[nodiscard]] bool empty() const { return x1 < x0 || y1 < y0; }
    [[nodiscard]] int width() const { return empty() ? 0 : x1 - x0 + 1; }
    [[nodiscard]] int height() const { return empty() ? 0 : y1 - y0 + 1; }
    [[nodiscard]] int longer() const { return std::max(width(), height()); }
    [[nodiscard]] long long area() const { return static_cast<long long>(width()) * height(); }
    [[nodiscard]] bool contains(int x, int y) const { return x >= x0 && x <= x1 && y >= y0 && y <= y1; }
    void add(int x, int y) {
      if (empty()) {
        *this = {x, y, x, y};
        return;
      }
      x0 = std::min(x0, x);
      y0 = std::min(y0, y);
      x1 = std::max(x1, x);
      y1 = std::max(y1, y);
    }
    void add(const box &other) {
      if (other.empty()) return;
      add(other.x0, other.y0);
      add(other.x1, other.y1);
    }
    bool operator==(const box &other) const {
      return empty() ? other.empty() : x0 == other.x0 && y0 == other.y0 && x1 == other.x1 && y1 == other.y1;
    }
  };

  [[nodiscard]] inline box intersect(const box &a, const box &b) {
    return {std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1), std::min(a.y1, b.y1)};
  }
  [[nodiscard]] inline bool overlaps(const box &a, const box &b) {
    return !a.empty() && !b.empty() && !intersect(a, b).empty();
  }
  [[nodiscard]] inline box grow(const box &b, int by) {
    return b.empty() ? b : box {b.x0 - by, b.y0 - by, b.x1 + by, b.y1 + by};
  }
  [[nodiscard]] inline box shift(const box &b, int dx, int dy) {
    return b.empty() ? b : box {b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy};
  }
  [[nodiscard]] inline box frame_box(unsigned width, unsigned height) {
    return {0, 0, int(width) - 1, int(height) - 1};
  }
  [[nodiscard]] inline double intersection_over_union(const box &a, const box &b) {
    const auto shared = overlaps(a, b) ? intersect(a, b).area() : 0;
    const auto total = a.area() + b.area() - shared;
    return total > 0 ? double(shared) / double(total) : 0.;
  }

  // ---------------------------------------------------------------- derived constants

  // The smallest square cell side giving at most max_cells cells: 35 px at
  // 3840x2160, 23 px at 2560x1440.
  [[nodiscard]] inline unsigned cell_side(unsigned width, unsigned height) {
    for (unsigned side = 1;; ++side) {
      if (std::uint64_t((width + side - 1) / side) * ((height + side - 1) / side) <= max_cells) return side;
    }
  }
  // The placement probe's central 75%: [floor(W/8), floor(7W/8)) by [floor(H/8), floor(7H/8)).
  [[nodiscard]] inline box central_rect(unsigned width, unsigned height) {
    return {int(width / 8), int(height / 8), int(7 * std::uint64_t(width) / 8) - 1, int(7 * std::uint64_t(height) / 8) - 1};
  }
  // Box quantization plus half a motion texel.
  [[nodiscard]] inline double epsilon_px(unsigned width, unsigned motion_width) {
    return 1 + .5 * double(width) / motion_width;
  }
  // One pixel of per-eye shift either side.
  [[nodiscard]] inline double delta_uv(unsigned width) {
    return 2. / width;
  }
  [[nodiscard]] inline double bin_uv(unsigned width) {
    return .25 / width;
  }
  // D(g): the largest target difference of two rigid targets with g unoccupied
  // columns between their texels in a shared row. The band's collar gives each
  // texel's bound to both neighbours (so one empty column joins nothing) and
  // its ramp adds 0.5/W per further column: U(x) - L(x) >= 0 at every x then.
  [[nodiscard]] inline double proximity_uv(int gap_columns, unsigned width) {
    return .5 * std::max(gap_columns - 1, 0) / width;
  }

  // ---------------------------------------------------------------- deterministic statistics

  // Integer histogram over [-C, C] with quarter-pixel bins; values beyond fall
  // in the end bins and non-finite values are skipped.
  class field_histogram {
  public:
    field_histogram(double bound, unsigned width):
        low_(-std::max(bound, 0.)),
        bin_(bin_uv(width)),
        counts_(std::max<std::size_t>(1, std::size_t(std::ceil(2 * std::max(bound, 0.) / bin_uv(width))))) {}

    void add(double value) {
      if (!std::isfinite(value)) return;
      const double at = std::floor((value - low_) / bin_);
      const auto last = double(counts_.size() - 1);
      ++counts_[std::size_t(std::clamp(at, 0., last))];
      ++total_;
    }
    [[nodiscard]] std::uint64_t total() const { return total_; }
    // The first bin whose cumulative count reaches ceil(percent% of the total).
    [[nodiscard]] std::size_t bin_at(std::uint32_t percent) const {
      const std::uint64_t target = std::max<std::uint64_t>(1, (std::uint64_t(percent) * total_ + 99) / 100);
      std::uint64_t seen = 0;
      for (std::size_t i = 0; i < counts_.size(); ++i) {
        seen += counts_[i];
        if (seen >= target) return i;
      }
      return counts_.size() - 1;
    }
    [[nodiscard]] double value(std::size_t bin) const { return low_ + (double(bin) + .5) * bin_; }
    [[nodiscard]] double percentile(std::uint32_t percent) const { return value(bin_at(percent)); }

  private:
    double low_, bin_;
    std::vector<std::uint64_t> counts_;
    std::uint64_t total_ {};
  };

  // Mean absolute deviation about the median (the lower middle for an even count).
  [[nodiscard]] inline double spread_about_median(std::vector<double> values) {
    if (values.empty()) return 0;
    auto middle = values.begin() + std::ptrdiff_t((values.size() - 1) / 2);
    std::nth_element(values.begin(), middle, values.end());
    const double median = *middle;
    double sum = 0;
    for (const double value : values) sum += std::abs(value - median);
    return sum / double(values.size());
  }

  [[nodiscard]] inline double median_of(std::vector<double> values) {
    if (values.empty()) return 0;
    auto middle = values.begin() + std::ptrdiff_t((values.size() - 1) / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
  }

  // ---------------------------------------------------------------- cells and labels

  struct cell {
    std::uint32_t occupancy {}, core {};  // texels with w > 0, and with w >= 1 (the P1 knee)
    box outline;                          // the bounds of its occupied texels
  };

  inline constexpr std::uint32_t no_label = 0xffffffffu;

  struct cell_grid {
    unsigned width {}, height {}, side {}, columns {}, rows {};
    std::vector<cell> cells;
    std::vector<std::uint32_t> label;  // the smallest cell index of its 8-connected component

    [[nodiscard]] std::uint32_t index_of(unsigned x, unsigned y) const { return (y / side) * columns + x / side; }
  };

  // The P1 pin weight as the mirror reads it: finite and positive, at most 1.
  [[nodiscard]] inline double pin_weight(float value) {
    return std::isfinite(value) && value > 0.f ? std::min(double(value), 1.) : 0.;
  }

  // SunshineUIAnchorCellsCS and SunshineUIAnchorLabelCS: per cell its occupancy,
  // core and outline; then 8-connected components of occupied cells labelled
  // by their smallest cell index, independent of visiting order.
  [[nodiscard]] inline cell_grid cells_of(const float *weight, unsigned width, unsigned height) {
    cell_grid grid;
    grid.width = width;
    grid.height = height;
    grid.side = cell_side(width, height);
    grid.columns = (width + grid.side - 1) / grid.side;
    grid.rows = (height + grid.side - 1) / grid.side;
    const std::size_t count = std::size_t(grid.columns) * grid.rows;
    grid.cells.assign(count, {});
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        const double w = pin_weight(weight[std::size_t(y) * width + x]);
        if (w <= 0) continue;
        auto &c = grid.cells[grid.index_of(x, y)];
        ++c.occupancy;
        c.core += w >= 1;
        c.outline.add(int(x), int(y));
      }
    }
    std::vector<std::uint32_t> parent(count);
    std::iota(parent.begin(), parent.end(), 0u);
    const auto find = [&](std::uint32_t at) {
      while (parent[at] != at) at = parent[at] = parent[parent[at]];
      return at;
    };
    const auto unite = [&](std::uint32_t a, std::uint32_t b) {
      a = find(a);
      b = find(b);
      if (a < b) parent[b] = a;
      else if (b < a) parent[a] = b;
    };
    for (unsigned cy = 0; cy < grid.rows; ++cy) {
      for (unsigned cx = 0; cx < grid.columns; ++cx) {
        const std::uint32_t at = cy * grid.columns + cx;
        if (!grid.cells[at].occupancy) continue;
        const int neighbours[4][2] {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}};
        for (const auto &n : neighbours) {
          const int nx = int(cx) + n[0], ny = int(cy) + n[1];
          if (nx < 0 || ny < 0 || nx >= int(grid.columns)) continue;
          const std::uint32_t other = std::uint32_t(ny) * grid.columns + std::uint32_t(nx);
          if (grid.cells[other].occupancy) unite(at, other);
        }
      }
    }
    grid.label.assign(count, no_label);
    for (std::uint32_t at = 0; at < count; ++at) {
      if (grid.cells[at].occupancy) grid.label[at] = find(at);
    }
    return grid;
  }

  // One connected component of occupied cells.
  struct element {
    std::uint32_t label {};
    std::vector<std::uint32_t> cells;
    box outline;
    std::uint64_t occupancy {}, core {};
  };

  [[nodiscard]] inline std::vector<element> elements_of(const cell_grid &grid) {
    std::map<std::uint32_t, element> by_label;
    for (std::uint32_t at = 0; at < grid.cells.size(); ++at) {
      if (grid.label[at] == no_label) continue;
      auto &e = by_label[grid.label[at]];
      e.label = grid.label[at];
      e.cells.push_back(at);
      e.outline.add(grid.cells[at].outline);
      e.occupancy += grid.cells[at].occupancy;
      e.core += grid.cells[at].core;
    }
    std::vector<element> result;
    result.reserve(by_label.size());
    for (auto &[label, e] : by_label) result.push_back(std::move(e));
    return result;
  }

  // Why an element is not a candidate, or null. Size and position never make an
  // element world; they only bound which elements gather evidence.
  [[nodiscard]] inline const char *candidate_refusal(const box &outline, std::uint64_t core, unsigned width, unsigned height) {
    if (!core) return "no_core";
    const auto center = central_rect(width, height);
    if (outline.x0 < center.x0 || outline.y0 < center.y0 || outline.x1 > center.x1 || outline.y1 > center.y1) return "edge";
    if (std::uint64_t(outline.longer()) * 4 > height) return "oversize";
    return nullptr;
  }

  // A full-frame decision: weight 1 on at least 99% of the central region.
  [[nodiscard]] inline bool central_full(const float *weight, unsigned width, unsigned height) {
    const auto center = central_rect(width, height);
    std::uint64_t full = 0;
    for (int y = center.y0; y <= center.y1; ++y) {
      for (int x = center.x0; x <= center.x1; ++x) full += pin_weight(weight[std::size_t(y) * width + std::size_t(x)]) >= 1;
    }
    return center.area() > 0 && full * 100 >= std::uint64_t(center.area()) * 99;
  }

  // ---------------------------------------------------------------- feasibility resolve

  // One element or sub-element for the resolve. Global items sit at pUI;
  // others take desired, projected into [lower, C] and within D(g) of every
  // placed or global neighbour that shares a row with them.
  struct resolve_item {
    bool global {};
    double desired {};
    double lower = sentinel;  // the floor this frame (a world element's), or none
    double order {};          // previous applied value: nearest placed first
    std::vector<box> outlines;  // its cells' outlines
  };

  struct resolve_result {
    std::vector<double> value;
    std::vector<std::uint8_t> demoted, floor_dropped;
    unsigned rounds {};
  };

  // The tightest D(g) between two items over every pair of their cell outlines
  // whose rows overlap; infinity when they share no row.
  [[nodiscard]] inline double proximity_between(const resolve_item &a, const resolve_item &b, unsigned width) {
    double result = std::numeric_limits<double>::infinity();
    for (const auto &p : a.outlines) {
      for (const auto &q : b.outlines) {
        if (p.empty() || q.empty() || p.y1 < q.y0 || q.y1 < p.y0) continue;
        const int distance = std::max(q.x0 - p.x1, p.x0 - q.x1);  // columns; at most 0 when they overlap
        result = std::min(result, proximity_uv(distance - 1, width));
      }
    }
    return result;
  }

  // Global UI is fixed at pUI; world elements are placed nearest first, each
  // projected into [floor, C] intersected with v +/- D(g) of placed or global
  // neighbours. On an empty interval the floor is dropped first; if it is still
  // empty the element is demoted to pUI and the resolve repeats.
  [[nodiscard]] inline resolve_result resolve(const std::vector<resolve_item> &items, double plane, double bound, unsigned width) {
    const std::size_t n = items.size();
    std::vector<double> near(n * n, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) near[i * n + j] = near[j * n + i] = proximity_between(items[i], items[j], width);
    }
    resolve_result out;
    out.value.assign(n, plane);
    out.demoted.assign(n, 0);
    out.floor_dropped.assign(n, 0);
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < n; ++i) {
      if (!items[i].global) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return items[a].order > items[b].order; });
    for (;;) {
      ++out.rounds;
      std::vector<std::uint8_t> placed(n, 0);
      for (std::size_t i = 0; i < n; ++i) {
        placed[i] = items[i].global || out.demoted[i];
        if (placed[i]) out.value[i] = plane;
        out.floor_dropped[i] = 0;
      }
      bool restart = false;
      for (const auto i : order) {
        if (out.demoted[i]) continue;
        double low = -bound, high = bound;
        for (std::size_t j = 0; j < n; ++j) {
          const double d = near[i * n + j];
          if (!placed[j] || !std::isfinite(d)) continue;
          low = std::max(low, out.value[j] - d);
          high = std::min(high, out.value[j] + d);
        }
        double floor_low = low;
        if (!is_sentinel(items[i].lower)) floor_low = std::max(low, items[i].lower);
        if (floor_low > high) {
          out.floor_dropped[i] = !is_sentinel(items[i].lower);
          floor_low = low;
        }
        if (floor_low > high) {
          out.demoted[i] = 1;
          out.value[i] = plane;
          restart = true;
          break;
        }
        out.value[i] = std::clamp(items[i].desired, floor_low, high);
        placed[i] = 1;
      }
      if (!restart) break;
    }
    return out;
  }

  // ---------------------------------------------------------------- motion

  struct flow {
    double x {}, y {};
  };

  [[nodiscard]] inline double norm_inf(const flow &f) {
    return std::max(std::abs(f.x), std::abs(f.y));
  }
  [[nodiscard]] inline flow minus(const flow &a, const flow &b) {
    return {a.x - b.x, a.y - b.y};
  }
  [[nodiscard]] inline bool finite(const flow &f) {
    return std::isfinite(f.x) && std::isfinite(f.y);
  }

  // A declared motion-vector scale: the vector times this scale is the UV
  // offset from a pixel in frame N to its position in frame N-1 (Streamline's
  // mvecScale, DLSS-G's MvecScaleX/Y).
  struct motion_semantics {
    double scale_x {}, scale_y {};
  };

  // The render grid of one real frame: inverse depth and, when the route
  // offers them, raw motion vectors.
  struct render_input {
    unsigned width {}, height {};
    const float *rho {};
    const std::array<float, 2> *motion {};
  };

  [[nodiscard]] inline int output_of(unsigned texel, unsigned render, unsigned output) {
    return int(std::floor((texel + .5) * output / render));
  }
  [[nodiscard]] inline int render_of(double pixel, unsigned output, unsigned render) {
    return int(std::floor((pixel + .5) * render / output));
  }

  // Decoded motion in output pixels from frame N-1 to frame N, per render texel.
  [[nodiscard]] inline std::vector<flow> decode_motion(const render_input &r, const motion_semantics &s, unsigned width, unsigned height) {
    std::vector<flow> result(std::size_t(r.width) * r.height, flow {sentinel, sentinel});
    if (!r.motion) return result;
    for (std::size_t i = 0; i < result.size(); ++i) {
      const auto &mv = r.motion[i];
      result[i] = {-double(mv[0]) * s.scale_x * width, -double(mv[1]) * s.scale_y * height};
    }
    return result;
  }

  // The quarter-resolution log-depth image of the motion check: the mean of
  // log rho over each 4x4 block, invalid (NaN) where any texel is at the far
  // plane or not finite.
  struct log_depth {
    unsigned width {}, height {};
    std::vector<double> value;
  };

  [[nodiscard]] inline log_depth quarter_log_depth(const render_input &r) {
    log_depth q;
    if (!r.rho) return q;
    q.width = (r.width + quarter - 1) / quarter;
    q.height = (r.height + quarter - 1) / quarter;
    q.value.assign(std::size_t(q.width) * q.height, sentinel);
    for (unsigned by = 0; by < q.height; ++by) {
      for (unsigned bx = 0; bx < q.width; ++bx) {
        double sum = 0;
        unsigned count = 0;
        bool valid = true;
        for (unsigned y = by * quarter; valid && y < std::min(r.height, (by + 1) * quarter); ++y) {
          for (unsigned x = bx * quarter; x < std::min(r.width, (bx + 1) * quarter); ++x) {
            const float rho = r.rho[std::size_t(y) * r.width + x];
            if (!std::isfinite(rho) || rho <= 0.f) {
              valid = false;
              break;
            }
            sum += std::log(double(rho));
            ++count;
          }
        }
        if (valid && count) q.value[std::size_t(by) * q.width + bx] = sum / count;
      }
    }
    return q;
  }

  // Bilinear log depth at quarter coordinates (block centres at integers); NaN
  // outside the image or when a tap is invalid.
  [[nodiscard]] inline double sample_log_depth(const log_depth &q, double x, double y) {
    if (q.value.empty() || !(x >= 0) || !(y >= 0) || x > q.width - 1. || y > q.height - 1.) return sentinel;
    const auto x0 = unsigned(std::floor(x)), y0 = unsigned(std::floor(y));
    const unsigned x1 = std::min(x0 + 1, q.width - 1), y1 = std::min(y0 + 1, q.height - 1);
    const double fx = x - x0, fy = y - y0;
    const auto at = [&](unsigned ix, unsigned iy) { return q.value[std::size_t(iy) * q.width + ix]; };
    const double a = at(x0, y0), b = at(x1, y0), c = at(x0, y1), d = at(x1, y1);
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;  // NaN propagates
  }

  enum class verdict : std::uint8_t { none, pass, fail };

  [[nodiscard]] inline const char *name(verdict value) {
    return value == verdict::pass ? "pass" : value == verdict::fail ? "fail" : "none";
  }

  // An informative check sample: quarter coordinates, frame N's log depth
  // there, and the decoded motion in quarter texels and in output pixels.
  struct check_probe {
    double x {}, y {}, current {}, fx {}, fy {}, px {}, py {};
  };

  struct check_result {
    verdict outcome = verdict::none;
    std::uint32_t samples {};
    std::vector<std::uint32_t> tested;     // per alternative: the samples it was compared on
    std::vector<double> decoded, alternatives;  // per alternative: both residuals on those samples
  };

  // Each alternative scales the decoded motion per axis: zero, either axis
  // negated, x2, x1/2, and the render-to-output ratio and its inverse (left out
  // when the ratio is one, where it equals the decoded motion).
  [[nodiscard]] inline std::vector<flow> check_alternatives(double ratio_x, double ratio_y) {
    std::vector<flow> result {{0, 0}, {-1, 1}, {1, -1}, {2, 2}, {.5, .5}};
    if (std::abs(ratio_x - 1) > 1e-9 || std::abs(ratio_y - 1) > 1e-9) {
      result.push_back({ratio_x, ratio_y});
      result.push_back({1 / ratio_x, 1 / ratio_y});
    }
    return result;
  }

  // How a hypothesis' residuals e = log rho_N(p) - log rho_N-1(p - motion)
  // are scored about their median (a depth-scale change does not enter).
  // median_deviation, the median of |e - median|, measures the bulk of the
  // image, so the minority of disoccluded samples, which no backward warp
  // explains, cannot outvote it. mean_deviation, the mean of |e - median|, is
  // the design's first draft: a correct declaration failed it in every frame
  // of a lock-on orbit, where the background disoccluded beside near objects
  // dominates the mean. It is kept for the test that shows this.
  enum class residual_score : std::uint8_t { median_deviation, mean_deviation };

  [[nodiscard]] inline double residual_of(std::vector<double> values, residual_score score) {
    if (values.empty()) return 0;
    if (score == residual_score::mean_deviation) return spread_about_median(std::move(values));
    const double median = median_of(values);
    for (auto &value : values) value = std::abs(value - median);
    return median_of(std::move(values));
  }

  // Compares the decoded motion with each alternative over the informative
  // samples valid for both. An alternative is tested only where it is a
  // different hypothesis: on at least n_min samples it moves by at least
  // 2 epsilon from the decoded motion (negating y is no alternative to a purely
  // horizontal motion). The check passes when at least one alternative was
  // tested and, for every tested one, the decoded residual is at most 1/factor
  // of the alternative's and the alternative's is positive; it fails when a
  // tested alternative's residual is lower than the decoded one's; otherwise,
  // or with fewer than n_min informative samples, it gives no verdict.
  [[nodiscard]] inline check_result judge_motion(const log_depth &previous, const std::vector<check_probe> &probes, const std::vector<flow> &alternatives,
      double eps, residual_score score = residual_score::median_deviation) {
    check_result out;
    std::vector<std::vector<double>> decoded(alternatives.size()), others(alternatives.size());
    std::vector<std::uint32_t> differing(alternatives.size(), 0);
    for (const auto &p : probes) {
      const double e = p.current - sample_log_depth(previous, p.x - p.fx, p.y - p.fy);
      if (!std::isfinite(e)) continue;
      ++out.samples;
      for (std::size_t k = 0; k < alternatives.size(); ++k) {
        const auto &a = alternatives[k];
        const double other = p.current - sample_log_depth(previous, p.x - a.x * p.fx, p.y - a.y * p.fy);
        if (!std::isfinite(other)) continue;
        decoded[k].push_back(e);
        others[k].push_back(other);
        differing[k] += std::max(std::abs(a.x - 1) * std::abs(p.px), std::abs(a.y - 1) * std::abs(p.py)) >= 2 * eps;
      }
    }
    if (out.samples < n_min) return out;
    bool pass = true, fail = false, tested = false;
    for (std::size_t k = 0; k < alternatives.size(); ++k) {
      out.tested.push_back(differing[k]);
      if (differing[k] < n_min) {
        out.decoded.push_back(sentinel);
        out.alternatives.push_back(sentinel);
        continue;
      }
      tested = true;
      const double mine = residual_of(std::move(decoded[k]), score), theirs = residual_of(std::move(others[k]), score);
      out.decoded.push_back(mine);
      out.alternatives.push_back(theirs);
      pass = pass && theirs > 0 && check_pass_factor * mine <= theirs;
      fail = fail || theirs < mine;
    }
    out.outcome = fail ? verdict::fail : tested && pass ? verdict::pass : verdict::none;
    return out;
  }

  // The frame's motion-vector check on a lattice of at most check_sample_cap
  // quarter texels, over the samples whose decoded motion is at least 2 epsilon.
  [[nodiscard]] inline check_result check_motion(const log_depth &previous, const log_depth &current, const std::vector<flow> &decoded,
      const render_input &r, unsigned width, unsigned height, residual_score score = residual_score::median_deviation) {
    if (previous.value.empty() || current.value.empty() || previous.width != current.width || previous.height != current.height ||
        decoded.size() != std::size_t(r.width) * r.height) return {};
    unsigned stride = 1;
    while (std::uint64_t((current.width + stride - 1) / stride) * ((current.height + stride - 1) / stride) > check_sample_cap) ++stride;
    const double eps = epsilon_px(width, r.width);
    std::vector<check_probe> probes;
    for (unsigned j = 0; j < current.height; j += stride) {
      for (unsigned i = 0; i < current.width; i += stride) {
        const double c = current.value[std::size_t(j) * current.width + i];
        if (!std::isfinite(c)) continue;
        const unsigned ri = std::min(i * quarter + quarter / 2, r.width - 1), rj = std::min(j * quarter + quarter / 2, r.height - 1);
        const auto f = decoded[std::size_t(rj) * r.width + ri];
        if (!finite(f) || norm_inf(f) < 2 * eps) continue;
        probes.push_back({double(i), double(j), c, f.x * r.width / width / quarter, f.y * r.height / height / quarter, f.x, f.y});
      }
    }
    return judge_motion(previous, probes, check_alternatives(double(r.width) / width, double(r.height) / height), eps, score);
  }

  // ---------------------------------------------------------------- samples

  // One motion texel of a reach or an owner box, read at its output pixel.
  struct sample {
    int x {}, y {};          // output pixel
    unsigned ri {}, rj {};   // render texel
    flow f;                  // decoded motion (NaN without motion vectors)
    double rho {}, h {};
  };

  // The motion texels whose output pixel lies in an area, evenly subsampled
  // (one stride for both axes) to at most cap samples.
  [[nodiscard]] inline std::vector<sample> lattice(const box &area, unsigned width, unsigned height, const render_input &r,
      const std::vector<flow> &decoded, const double *field, std::uint32_t cap) {
    std::vector<sample> result;
    const box clipped = intersect(area, frame_box(width, height));
    if (clipped.empty()) return result;
    const auto first = [&](int pixel, unsigned render, unsigned output) {
      unsigned i = unsigned(std::max(0, render_of(pixel, output, render) - 1));
      while (i < render && output_of(i, render, output) < pixel) ++i;
      return i;
    };
    const unsigned i0 = first(clipped.x0, r.width, width), j0 = first(clipped.y0, r.height, height);
    unsigned i1 = i0, j1 = j0;
    while (i1 < r.width && output_of(i1, r.width, width) <= clipped.x1) ++i1;
    while (j1 < r.height && output_of(j1, r.height, height) <= clipped.y1) ++j1;
    if (i1 <= i0 || j1 <= j0) return result;
    unsigned stride = 1;
    while (std::uint64_t((i1 - i0 + stride - 1) / stride) * ((j1 - j0 + stride - 1) / stride) > cap) ++stride;
    for (unsigned j = j0; j < j1; j += stride) {
      for (unsigned i = i0; i < i1; i += stride) {
        sample s;
        s.ri = i;
        s.rj = j;
        s.x = output_of(i, r.width, width);
        s.y = output_of(j, r.height, height);
        const std::size_t at = std::size_t(j) * r.width + i;
        s.f = decoded.size() > at ? decoded[at] : flow {sentinel, sentinel};
        s.rho = r.rho ? double(r.rho[at]) : sentinel;
        s.h = field[std::size_t(s.y) * width + std::size_t(s.x)];
        result.push_back(s);
      }
    }
    return result;
  }

  // The flow line of the samples that do not co-move: split at the median rho,
  // the nearer and farther halves' median flows and depths give f(rho) = a + b rho.
  struct flow_line {
    bool discriminates {};
    flow a;
  };

  [[nodiscard]] inline flow_line fit_flow_line(std::vector<const sample *> rest, double eps) {
    flow_line out;
    if (rest.size() < 2) return out;
    std::stable_sort(rest.begin(), rest.end(), [](const sample *p, const sample *q) { return p->rho < q->rho; });
    const std::size_t half = rest.size() / 2;
    const auto medians = [&](std::size_t begin, std::size_t end) {
      std::vector<double> x, y, rho;
      for (std::size_t i = begin; i < end; ++i) {
        x.push_back(rest[i]->f.x);
        y.push_back(rest[i]->f.y);
        rho.push_back(rest[i]->rho);
      }
      return std::array<double, 3> {median_of(x), median_of(y), median_of(rho)};
    };
    const auto far = medians(0, half), near = medians(rest.size() - half, rest.size());
    if (!(near[2] > far[2])) return out;
    const flow f_far {far[0], far[1]}, f_near {near[0], near[1]};
    out.discriminates = norm_inf(minus(f_near, f_far)) >= 2 * eps;
    const double run = near[2] - far[2];
    const flow b {(f_near.x - f_far.x) / run, (f_near.y - f_far.y) / run};
    out.a = {f_far.x - b.x * far[2], f_far.y - b.y * far[2]};
    return out;
  }

  // No scalar s maps (d1, a1) onto (d2, a2) within tol per component.
  [[nodiscard]] inline bool distinct(const std::array<double, 4> &first, const std::array<double, 4> &second, double tol) {
    double low = -std::numeric_limits<double>::infinity(), high = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 4; ++i) {
      if (first[i] == 0) {
        if (std::abs(second[i]) > tol) return true;
        continue;
      }
      double a = (second[i] - tol) / first[i], b = (second[i] + tol) / first[i];
      if (a > b) std::swap(a, b);
      low = std::max(low, a);
      high = std::min(high, b);
    }
    return low > high;
  }

  // ---------------------------------------------------------------- tracks and frames

  enum class track_class : std::uint8_t { global, pending, world };

  [[nodiscard]] inline const char *name(track_class value) {
    return value == track_class::world ? "world" : value == track_class::pending ? "pending" : "global";
  }

  struct track {
    bool live {};
    track_class cls = track_class::global;
    box outline;                       // the last measured outline
    std::uint64_t outline_sequence {}; // the real frame that measured it
    int motion_x {}, motion_y {};      // its last change, for the prediction
    double goal = sentinel;
    box owner;                         // the owner box, relative to the outline's corner
    double applied = sentinel;
    std::array<double, 4> first {};    // the pending record (d_E, a)
    std::uint64_t first_ms {}, support_ms {};
    bool exiting {};
    std::uint64_t exit_since {};
    bool unmatched {};
    std::uint64_t unmatched_since {};
    bool follow_ok {};
    bool floor_confirmed {}, floor_approaching {};
    double floor_value {}, floor_candidate {};
    std::uint64_t floor_since {};
    double floor_observed = sentinel;
  };

  // The CPU's four pair conditions (Frame identity): the route pairs mask,
  // depth and motion of one real frame; the winner's snapshots differ and the
  // depth is the next capture of the same source; the semantics are present
  // and admissible; the depth is current and fresh. Every one defaults to
  // false: a missing proof refuses the pair.
  struct pair_proof {
    bool route {}, consecutive {}, semantics {}, fresh {};
  };

  struct frame {
    std::uint64_t now_ms {}, sequence {};  // sequence counts real frames
    unsigned width {}, height {};
    const float *weight {};                // P1 pin weights of the selected mask
    const double *field {};                // the unpinned displayed field h
    double plane {}, bound {};             // pUI and C
    render_input render;
    motion_semantics semantics;
    pair_proof proof;
    std::uint64_t scope {}, winner {}, route {};  // scope, winner kind and signature, route signature
    std::uint64_t frame_index {};          // Streamline frame index, 0 when absent
    bool reused_decision {};  // T1 grace: this real frame reuses the previous decision
    bool full_frame {};       // sources 6 or 8
    bool hidden {};           // a held hidden verdict (H1)
    bool hold_reset {};       // per_frame_hold_reset or held.none
    bool inactive {};
    bool camera_reset {};     // camera or NGX reset
    bool revision {};         // an observation revision
  };

  // An element, or one track's sub-element of a merged component.
  struct part {
    box outline;
    std::vector<std::uint32_t> cells;
    std::uint64_t occupancy {}, core {};
    std::uint32_t component {};
    int track = -1;
    bool candidate {}, merged {};
    const char *refusal {};
    double inherit = sentinel;  // a split part's parent value
    track_class cls = track_class::global;
    double value = sentinel;
    const char *outcome = "none";
  };

  struct result {
    cell_grid grid;
    std::vector<part> parts;
    std::vector<double> cell_target;         // per cell; the sentinel is pUI
    std::vector<std::uint8_t> group_flags;   // per 8-row pin group: rows with a non-sentinel target
    verdict check = verdict::none;
    const char *pair = "none";               // valid, or the first refusal
    std::uint32_t world {}, pending {};
  };

  struct counters {
    // steps: applied values that moved faster than the approach rate (a
    // constraint appearing abruptly: a projection or a demotion).
    std::uint64_t frames {}, pairs {}, valid_pairs {}, demotions {}, floor_drops {}, floor_lifts {}, transitions {}, steps {};
    std::map<std::string, std::uint64_t> refusals, verdicts, outcomes, entries, exits, drops;
    double travel_uv {}, world_element_ms {};
    std::uint64_t oldest_support_ms {};
  };

  // Per texel with w > 0: its cell's target; the sentinel elsewhere.
  [[nodiscard]] inline std::vector<double> texel_targets(const result &r, const float *weight) {
    const auto &g = r.grid;
    std::vector<double> out(std::size_t(g.width) * g.height, sentinel);
    if (r.cell_target.size() != g.cells.size()) return out;
    for (unsigned y = 0; y < g.height; ++y) {
      for (unsigned x = 0; x < g.width; ++x) {
        const std::size_t at = std::size_t(y) * g.width + x;
        if (pin_weight(weight[at]) > 0) out[at] = r.cell_target[g.index_of(x, y)];
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- the classifier

  // One renderer's anchor state over armed real frames. Generated Presents
  // never call step(); they show last(), the real frame's targets.
  class classifier {
  public:
    [[nodiscard]] const counters &stats() const { return counters_; }
    [[nodiscard]] const std::array<track, max_tracks> &tracks() const { return tracks_; }
    [[nodiscard]] const result &last() const { return last_; }

    const result &step(const frame &f) {
      validate(f);
      ++counters_.frames;
      const unsigned width = f.width, height = f.height;
      const std::uint64_t now = f.now_ms;
      const std::uint64_t dt = have_clock_ && now >= clock_ ? std::min(now - clock_, ui_adaptive::max_age_ms) : 0;
      clock_ = now;
      have_clock_ = true;
      const double eps = epsilon_px(width, f.render.width), delta = delta_uv(width);
      result out;

      // Drop all tracks: a scope change, a full-frame decision, a held hidden verdict.
      const char *drop = have_scope_ && f.scope != scope_ ? "scope" :
                         f.full_frame || central_full(f.weight, width, height) ? "full_frame" :
                         f.hidden ? "hidden" : nullptr;
      if (drop) {
        if (have_scope_ && f.scope != scope_) gates_.clear();
        for (auto &t : tracks_) t = {};
        ++counters_.drops[drop];
        out.grid = cells_of(f.weight, width, height);
        out.cell_target.assign(out.grid.cells.size(), sentinel);
        out.group_flags.assign((height + pin_group_rows - 1) / pin_group_rows, 0);
        out.pair = drop;
        ++counters_.refusals[drop];
        forget_history(f);
        last_ = out;
        last_parts_.clear();
        return last_;
      }
      scope_ = f.scope;
      have_scope_ = true;

      // Return every track to global, refusing motion across the boundary.
      const char *boundary = have_winner_ && f.winner != winner_ ? "winner" : f.hold_reset ? "hold_reset" :
                             f.inactive ? "inactive" : f.camera_reset ? "camera_reset" : nullptr;
      winner_ = f.winner;
      have_winner_ = true;
      if (boundary) return_all(boundary, now);

      // Segmentation and association, or the reused decision's elements.
      const bool reuse = f.reused_decision && !last_parts_.empty();
      std::vector<double> before(max_tracks);
      for (std::size_t t = 0; t < max_tracks; ++t) before[t] = tracks_[t].applied;
      std::vector<box> previous_outline(max_tracks);
      std::vector<std::uint64_t> previous_sequence(max_tracks);
      for (std::size_t t = 0; t < max_tracks; ++t) {
        previous_outline[t] = tracks_[t].outline;
        previous_sequence[t] = tracks_[t].outline_sequence;
      }
      std::vector<std::uint8_t> created(max_tracks, 0);
      if (reuse) {
        out.grid = last_.grid;
        out.parts = last_parts_;
        for (auto &p : out.parts) {
          if (p.track >= 0 && !tracks_[std::size_t(p.track)].live) p.track = -1;
          p.outcome = "reused";
        }
      } else {
        out.grid = cells_of(f.weight, width, height);
        out.parts = associate(out.grid, f, created);
      }

      // Pair validity: the first refusal is counted.
      const auto decoded = decode_motion(f.render, f.semantics, width, height);
      const auto current_q = quarter_log_depth(f.render);
      const char *pair = reuse || f.reused_decision ? "reused" : boundary ? boundary : f.revision ? "revision" :
                         !table_fresh_ || f.sequence != table_sequence_ + 1 ? "not_consecutive" :
                         f.frame_index && previous_index_ && f.frame_index != previous_index_ + 1 ? "frame_index" :
                         !f.proof.route ? "route" : !f.proof.consecutive ? "consecutive" :
                         !f.proof.semantics ? "semantics" : !f.proof.fresh ? "fresh" :
                         !f.render.motion || !f.render.rho ? "no_motion" : nullptr;
      if (!pair) {
        ++counters_.pairs;
        out.check = check_motion(previous_q_, current_q, decoded, f.render, width, height).outcome;
        ++counters_.verdicts[name(out.check)];
        auto &gate = gates_[f.route];
        if (out.check == verdict::pass) gate = true;
        if (out.check == verdict::fail) {
          gate = false;
          return_all("check_failed", now);
          pair = "check_failed";
        } else if (!gate) {
          pair = "gate_closed";
        }
      }
      out.pair = pair ? pair : "valid";
      if (pair) ++counters_.refusals[pair];
      else ++counters_.valid_pairs;

      // Motion evidence and classification on a valid pair.
      for (auto &p : out.parts) {
        if (p.track < 0) continue;
        auto &t = tracks_[std::size_t(p.track)];
        if (!p.candidate && t.cls != track_class::global) to_global(t, "not_candidate", now);
        if (pair || !p.candidate || created[std::size_t(p.track)]) continue;
        if (p.merged) {
          p.outcome = "merged";
          ++counters_.outcomes[p.outcome];
          continue;
        }
        const auto &prior = previous_outline[std::size_t(p.track)];
        if (prior.empty() || previous_sequence[std::size_t(p.track)] + 1 != f.sequence) {
          p.outcome = "no_previous_box";
          ++counters_.outcomes[p.outcome];
          continue;
        }
        classify(t, p, prior, f, decoded, current_q, eps, delta);
        ++counters_.outcomes[p.outcome];
      }

      // Track bookkeeping: outlines, support bound, unmatched tracks.
      std::vector<int> part_of(max_tracks, -1);
      for (std::size_t i = 0; i < out.parts.size(); ++i) {
        if (out.parts[i].track >= 0) part_of[std::size_t(out.parts[i].track)] = int(i);
      }
      for (std::size_t ti = 0; ti < max_tracks; ++ti) {
        auto &t = tracks_[ti];
        if (!t.live) continue;
        if (part_of[ti] >= 0 && !reuse) {
          const auto &p = out.parts[std::size_t(part_of[ti])];
          const auto &prior = previous_outline[ti];
          if (!created[ti] && !prior.empty() && previous_sequence[ti] + 1 == f.sequence) {
            t.motion_x = int(std::lround((p.outline.x0 - prior.x0 + p.outline.x1 - prior.x1) / 2.));
            t.motion_y = int(std::lround((p.outline.y0 - prior.y0 + p.outline.y1 - prior.y1) / 2.));
          } else {
            t.motion_x = t.motion_y = 0;
          }
          t.outline = p.outline;
          t.outline_sequence = f.sequence;
          t.unmatched = false;
        } else if (part_of[ti] < 0) {
          // An unmatched track waits up to the support bound for its element.
          if (!t.unmatched) {
            t.unmatched = true;
            t.unmatched_since = now;
          }
          if (now - std::min(now, t.unmatched_since) > ui_adaptive::release_dwell_ms) {
            if (t.cls == track_class::world) ++counters_.transitions;
            ++counters_.drops["unmatched"];
            t = {};
            continue;
          }
        }
        if (t.cls != track_class::global && now - std::min(now, t.support_ms) > ui_adaptive::release_dwell_ms) to_global(t, "support", now);
      }

      // Follow, floor and the applied value's ramp; then the feasibility resolve.
      const double step_retreat = ui_adaptive::retreat_rate_uv_per_second * double(dt) / 1000;
      const double step_approach = ui_adaptive::approach_rate_uv_per_second * double(dt) / 1000;
      std::vector<resolve_item> items(out.parts.size());
      for (std::size_t i = 0; i < out.parts.size(); ++i) {
        auto &p = out.parts[i];
        auto &item = items[i];
        for (const auto c : p.cells) item.outlines.push_back(out.grid.cells[c].outline);
        item.global = true;
        if (p.track < 0) continue;
        auto &t = tracks_[std::size_t(p.track)];
        if (t.cls != track_class::global) follow(t, p, f, decoded, delta);
        if (t.cls == track_class::world) observe_floor(t, p, out.grid, f, now);
        else clear_floor(t);
        double v = t.applied;
        if (t.cls == track_class::world && is_sentinel(v)) v = f.plane;
        if (is_sentinel(v)) continue;
        const double target = t.cls == track_class::world && t.follow_ok ? t.goal : f.plane;
        double next = v + std::clamp(target - v, -step_retreat, step_retreat);
        if (t.cls == track_class::world && t.floor_confirmed) {
          const double floor = t.floor_value;
          next = v >= floor ? std::max(next, floor) : std::min(floor, std::max(next, v + step_approach));
          if (v < floor) ++counters_.floor_lifts;
        }
        next = std::clamp(next, -f.bound, f.bound);
        if (t.cls != track_class::world && std::abs(next - f.plane) <= step_retreat) {
          t.applied = sentinel;
          continue;
        }
        item.global = false;
        item.desired = next;
        item.order = is_sentinel(before[std::size_t(p.track)]) ? f.plane : before[std::size_t(p.track)];
        if (t.cls == track_class::world && t.floor_confirmed) item.lower = std::min(t.floor_value, next);
      }
      const auto resolved = resolve(items, f.plane, f.bound, width);
      out.cell_target.assign(out.grid.cells.size(), sentinel);
      out.group_flags.assign((height + pin_group_rows - 1) / pin_group_rows, 0);
      for (std::size_t i = 0; i < out.parts.size(); ++i) {
        auto &p = out.parts[i];
        if (p.track >= 0) {
          auto &t = tracks_[std::size_t(p.track)];
          p.cls = t.cls;
          const double old = before[std::size_t(p.track)];
          if (!items[i].global) {
            t.applied = resolved.value[i];
            counters_.demotions += resolved.demoted[i];
            counters_.floor_drops += resolved.floor_dropped[i];
            counters_.steps += std::abs(t.applied - (is_sentinel(old) ? f.plane : old)) > step_approach + 1e-12;
          }
          if (t.cls == track_class::world && !is_sentinel(old) && !is_sentinel(t.applied)) counters_.travel_uv += std::abs(t.applied - old);
          if (t.cls == track_class::world) {
            counters_.world_element_ms += double(dt);
            ++out.world;
          }
          out.pending += t.cls == track_class::pending;
        }
        p.value = items[i].global ? sentinel : resolved.value[i];
        if (is_sentinel(p.value)) continue;
        for (const auto c : p.cells) {
          out.cell_target[c] = p.value;
          const auto &o = out.grid.cells[c].outline;
          for (int g = o.y0 / int(pin_group_rows); g <= o.y1 / int(pin_group_rows); ++g) out.group_flags[std::size_t(g)] = 1;
        }
      }
      counters_.oldest_support_ms = 0;
      for (const auto &t : tracks_) {
        if (t.live && t.cls != track_class::global) counters_.oldest_support_ms = std::max(counters_.oldest_support_ms, now - std::min(now, t.support_ms));
      }

      // History for the next pair.
      previous_q_ = current_q;
      previous_decoded_ = decoded;
      decoded_sequence_ = f.sequence;
      have_decoded_ = !reuse && f.render.motion;
      table_sequence_ = f.sequence;
      table_fresh_ = !f.reused_decision;
      previous_index_ = f.frame_index;
      last_parts_ = out.parts;
      last_ = std::move(out);
      return last_;
    }

  private:
    static void validate(const frame &f) {
      if (!f.width || !f.height || f.width > 16384 || f.height > 16384 || !f.weight || !f.field)
        throw std::invalid_argument("ui_anchor frame needs an extent, a weight mask and a field");
      if (!f.render.width || !f.render.height || (f.render.motion && !f.render.rho))
        throw std::invalid_argument("ui_anchor frame needs a render grid, and depth with its motion");
      if (!std::isfinite(f.plane) || !std::isfinite(f.bound) || f.bound < 0)
        throw std::invalid_argument("ui_anchor frame needs a finite plane and bound");
    }

    void forget_history(const frame &f) {
      previous_q_ = {};
      previous_decoded_.clear();
      have_decoded_ = false;
      table_sequence_ = f.sequence;
      table_fresh_ = false;
      previous_index_ = f.frame_index;
      scope_ = f.scope;
      have_scope_ = true;
      winner_ = f.winner;
      have_winner_ = true;
    }

    void to_global(track &t, const char *reason, std::uint64_t now) {
      if (t.cls == track_class::global) return;
      if (t.cls == track_class::world) {
        ++counters_.exits[reason];
        ++counters_.transitions;
      }
      (void) now;
      t.cls = track_class::global;
      t.exiting = false;
      t.goal = sentinel;
      t.owner = {};
      t.follow_ok = false;
      clear_floor(t);
    }

    void return_all(const char *reason, std::uint64_t now) {
      for (auto &t : tracks_) {
        if (t.live) to_global(t, reason, now);
      }
    }

    static void clear_floor(track &t) {
      t.floor_confirmed = t.floor_approaching = false;
      t.floor_observed = sentinel;
    }

    // Association: each track predicts its box (last box shifted by its last
    // motion, grown by one cell). A component under no predicted box is new;
    // under one, it goes to that track. A component under several (touching
    // bars) is split: its cells under exactly one box form that track's
    // sub-element, and cells under several or none join the member whose
    // applied value is nearest pUI (the sentinel first, so a touching HUD
    // element never takes another's world target). A track claimed by several
    // components keeps the part with the best intersection over union with its
    // prediction; the others start global tracks inheriting its applied value.
    std::vector<part> associate(const cell_grid &grid, const frame &f, std::vector<std::uint8_t> &created) {
      std::vector<box> predicted(max_tracks);
      for (std::size_t t = 0; t < max_tracks; ++t) {
        if (tracks_[t].live) predicted[t] = grow(shift(tracks_[t].outline, tracks_[t].motion_x, tracks_[t].motion_y), int(grid.side));
      }
      const auto distance_to_plane = [&](std::size_t t) {
        return is_sentinel(tracks_[t].applied) ? 0. : std::abs(tracks_[t].applied - f.plane);
      };
      std::vector<part> parts;
      const auto make = [&](std::uint32_t component, std::vector<std::uint32_t> cells, int track) {
        part p;
        p.component = component;
        p.track = track;
        for (const auto c : cells) {
          p.outline.add(grid.cells[c].outline);
          p.occupancy += grid.cells[c].occupancy;
          p.core += grid.cells[c].core;
        }
        p.cells = std::move(cells);
        return p;
      };
      for (const auto &e : elements_of(grid)) {
        std::vector<std::size_t> members;
        for (std::size_t t = 0; t < max_tracks; ++t) {
          if (!tracks_[t].live) continue;
          for (const auto c : e.cells) {
            if (overlaps(grid.cells[c].outline, predicted[t])) {
              members.push_back(t);
              break;
            }
          }
        }
        if (members.size() <= 1) {
          parts.push_back(make(e.label, e.cells, members.empty() ? -1 : int(members.front())));
          continue;
        }
        std::map<std::size_t, std::vector<std::uint32_t>> owned;
        for (const auto c : e.cells) {
          std::vector<std::size_t> inside;
          for (const auto t : members) {
            if (overlaps(grid.cells[c].outline, predicted[t])) inside.push_back(t);
          }
          std::size_t pick = inside.size() == 1 ? inside.front() : members.front();
          if (inside.size() != 1) {
            for (const auto t : members) {
              if (distance_to_plane(t) < distance_to_plane(pick)) pick = t;
            }
          }
          owned[pick].push_back(c);
        }
        const bool merged = owned.size() >= 2;
        for (auto &[t, cells] : owned) {
          auto p = make(e.label, std::move(cells), int(t));
          p.merged = merged;
          parts.push_back(std::move(p));
        }
      }
      // A track claimed by several components keeps its best match.
      for (std::size_t t = 0; t < max_tracks; ++t) {
        int best = -1;
        double best_iou = -1;
        for (std::size_t i = 0; i < parts.size(); ++i) {
          if (parts[i].track != int(t)) continue;
          const double iou = intersection_over_union(parts[i].outline, predicted[t]);
          if (iou > best_iou) {
            best_iou = iou;
            best = int(i);
          }
        }
        for (std::size_t i = 0; i < parts.size(); ++i) {
          if (parts[i].track == int(t) && int(i) != best) {
            parts[i].track = -1;
            parts[i].inherit = tracks_[t].applied;
          }
        }
      }
      // Candidates; new candidates take free rows, tracked parts first.
      std::stable_sort(parts.begin(), parts.end(), [](const part &a, const part &b) {
        return (a.track >= 0) > (b.track >= 0) || ((a.track >= 0) == (b.track >= 0) && a.component < b.component);
      });
      for (auto &p : parts) {
        p.refusal = candidate_refusal(p.outline, p.core, f.width, f.height);
        p.candidate = !p.refusal;
        if (p.track >= 0 || !p.candidate) continue;
        std::size_t slot = 0;
        while (slot < max_tracks && tracks_[slot].live) ++slot;
        if (slot == max_tracks) {
          p.candidate = false;
          p.refusal = "track_capacity";
          continue;
        }
        auto &t = tracks_[slot];
        t = {};
        t.live = true;
        t.applied = p.inherit;
        t.outline = p.outline;
        t.outline_sequence = f.sequence;
        p.track = int(slot);
        created[slot] = 1;
      }
      return parts;
    }

    // Element motion, scene samples, evidence and the class transitions.
    void classify(track &t, part &p, const box &prior, const frame &f, const std::vector<flow> &decoded, const log_depth &current_q,
        double eps, double delta) {
      const unsigned width = f.width, height = f.height;
      // Rigid only if left and right edges moved alike within 1 px, and top and bottom too.
      const int left = p.outline.x0 - prior.x0, right = p.outline.x1 - prior.x1;
      const int top = p.outline.y0 - prior.y0, bottom = p.outline.y1 - prior.y1;
      if (std::abs(left - right) > 1 || std::abs(top - bottom) > 1) {
        p.outcome = "not_rigid";
        return;
      }
      const flow d {(left + right) / 2., (top + bottom) / 2.};
      const bool still = norm_inf(d) <= 1, moving = norm_inf(d) >= 2 * eps;
      const box reach = grow(p.outline, p.outline.longer());
      const auto samples = lattice(reach, width, height, f.render, decoded, f.field, reach_sample_cap);

      // Unexplained: zero motion explains the reach's depth change better than the decoded motion.
      {
        std::vector<check_probe> probes;
        for (const auto &s : samples) {
          if (!finite(s.f) || norm_inf(s.f) < 2 * eps) continue;
          const double qx = (s.ri + .5) / quarter - .5, qy = (s.rj + .5) / quarter - .5;
          const double c = sample_log_depth(current_q, qx, qy);
          if (!std::isfinite(c)) continue;
          probes.push_back({qx, qy, c, s.f.x * f.render.width / width / quarter, s.f.y * f.render.height / height / quarter, s.f.x, s.f.y});
        }
        if (judge_motion(previous_q_, probes, {{0, 0}}, eps).outcome == verdict::fail) {
          p.outcome = "unexplained";
          return;
        }
      }

      std::vector<const sample *> matching, rest;
      for (const auto &s : samples) {
        if (!finite(s.f) || !std::isfinite(s.h)) continue;
        const double off = norm_inf(minus(s.f, d));
        if (off <= eps) matching.push_back(&s);
        else if (off > 2 * eps && std::isfinite(s.rho)) rest.push_back(&s);
      }
      const auto line = rest.size() >= n_min ? fit_flow_line(rest, eps) : flow_line {};
      const bool discriminating = rest.size() >= n_min && line.discriminates;

      // World evidence.
      bool world = false;
      double world_goal = sentinel;
      box world_owner;
      if (moving && matching.size() >= n_min && discriminating) {
        field_histogram hist(f.bound, width);
        for (const auto *s : matching) hist.add(s->h);
        if (hist.bin_at(90) - hist.bin_at(10) <= delta_bins) {
          world = true;
          world_goal = hist.percentile(50);
          for (const auto *s : matching) world_owner.add(s->x, s->y);
        }
      }
      const std::array<double, 4> record {d.x, d.y, line.a.x, line.a.y};

      // Consistency and contradiction at the track's goal.
      bool consistent = false, contradicted = false;
      double consistent_goal = sentinel;
      box consistent_owner;
      if (t.cls != track_class::global && !is_sentinel(t.goal)) {
        field_histogram hist(f.bound, width);
        if (still || moving) {
          for (const auto *s : matching) {
            if (std::abs(s->h - t.goal) <= delta) {
              hist.add(s->h);
              consistent_owner.add(s->x, s->y);
            }
          }
          consistent = hist.total() >= n_min;
          if (consistent) consistent_goal = hist.percentile(50);
        }
        if (!consistent) {
          std::uint32_t refuting = 0, agreeing = 0;
          const bool previous_flow = have_decoded_ && decoded_sequence_ + 1 == f.sequence;
          for (const auto &s : samples) {
            if (!finite(s.f) || !std::isfinite(s.h) || std::abs(s.h - t.goal) > delta) continue;
            const bool refutes_now = norm_inf(minus(s.f, d)) > 2 * eps, agrees_now = norm_inf(minus(s.f, d)) <= eps;
            bool refutes_before = true, agrees_before = false;
            if (previous_flow) {
              const int ri = render_of(s.x - s.f.x, width, f.render.width), rj = render_of(s.y - s.f.y, height, f.render.height);
              if (ri >= 0 && rj >= 0 && ri < int(f.render.width) && rj < int(f.render.height)) {
                const auto &before = previous_decoded_[std::size_t(rj) * f.render.width + std::size_t(ri)];
                if (finite(before)) {
                  refutes_before = norm_inf(minus(before, d)) > 2 * eps;
                  agrees_before = norm_inf(minus(before, d)) <= eps;
                }
              }
            }
            refuting += refutes_now && refutes_before;
            agreeing += agrees_now || agrees_before;
          }
          contradicted = refuting >= n_min && agreeing < n_min;
        }
      }
      const auto now = f.now_ms;
      const auto relative = [&](const box &owner) { return shift(owner, -p.outline.x0, -p.outline.y0); };
      p.outcome = world ? "world" : consistent ? (discriminating ? "discriminating" : "consistency") : contradicted ? "contradiction" : "none";

      switch (t.cls) {
        case track_class::global:
          if (world) {
            t.cls = track_class::pending;
            t.first = record;
            t.first_ms = t.support_ms = now;
            t.goal = world_goal;
            t.owner = relative(world_owner);
          }
          break;
        case track_class::pending:
          if (contradicted) {
            t.cls = track_class::global;
            t.goal = sentinel;
            t.owner = {};
            ++counters_.outcomes["pending_contradicted"];
          } else if (world) {
            t.support_ms = now;
            if (std::abs(world_goal - t.goal) > delta) {
              t.first = record;  // a disagreeing goal restarts pending
              t.first_ms = now;
              t.goal = world_goal;
              t.owner = relative(world_owner);
              ++counters_.outcomes["pending_restarted"];
            } else if (now - t.first_ms >= ui_adaptive::approach_dwell_ms && distinct(t.first, record, 2 * eps)) {
              t.cls = track_class::world;
              t.goal = world_goal;
              t.owner = relative(world_owner);
              t.exiting = false;
              ++counters_.entries["world_evidence"];
              ++counters_.transitions;
            }
          } else if (consistent) {
            t.goal = consistent_goal;
            if (discriminating) t.support_ms = now;
          }
          break;
        case track_class::world:
          if (consistent) {
            t.exiting = false;
            t.goal = consistent_goal;
            if (discriminating) {
              t.support_ms = now;
              t.owner = relative(consistent_owner);
            }
          } else if (contradicted) {
            if (!t.exiting) {
              t.exiting = true;
              t.exit_since = now;
            } else if (now - t.exit_since >= ui_adaptive::approach_dwell_ms) {
              to_global(t, "contradiction", now);
            }
          }
          break;
      }
    }

    // The goal moves to the median h of owner-box samples within delta of it,
    // when there are n_min; the target is the goal only while this succeeds.
    void follow(track &t, const part &p, const frame &f, const std::vector<flow> &decoded, double delta) {
      t.follow_ok = false;
      if (is_sentinel(t.goal) || t.owner.empty()) return;
      const box owner = shift(t.owner, p.outline.x0, p.outline.y0);
      field_histogram hist(f.bound, f.width);
      for (const auto &s : lattice(owner, f.width, f.height, f.render, decoded, f.field, reach_sample_cap)) {
        if (std::isfinite(s.h) && std::abs(s.h - t.goal) <= delta) hist.add(s.h);
      }
      if (hist.total() < n_min) return;
      t.goal = hist.percentile(50);
      t.follow_ok = true;
    }

    // F_E: the 80th percentile of h over the element's core plus the probe's
    // margin. A floor above the confirmed one acts only after persisting for
    // approach_dwell_ms (its least forward value through the dwell); a lower
    // one replaces it at once and the value follows through the ramp.
    void observe_floor(track &t, const part &p, const cell_grid &grid, const frame &f, std::uint64_t now) {
      field_histogram hist(f.bound, f.width);
      for (const auto c : p.cells) {
        const auto &o = grid.cells[c].outline;
        for (int y = o.y0; y <= o.y1; ++y) {
          for (int x = o.x0; x <= o.x1; ++x) {
            const std::size_t at = std::size_t(y) * f.width + std::size_t(x);
            if (pin_weight(f.weight[at]) >= 1 && grid.index_of(unsigned(x), unsigned(y)) == c) hist.add(f.field[at]);
          }
        }
      }
      if (!hist.total()) return;
      const double floor = hist.percentile(floor_percent) + probe_margin * f.bound;
      t.floor_observed = floor;
      const double held = t.floor_confirmed ? t.floor_value : (is_sentinel(t.applied) ? f.plane : t.applied);
      if (t.floor_confirmed && floor <= t.floor_value) {
        t.floor_value = floor;
        t.floor_approaching = false;
      } else if (floor > held) {
        if (!t.floor_approaching) {
          t.floor_approaching = true;
          t.floor_since = now;
          t.floor_candidate = floor;
        } else {
          t.floor_candidate = std::min(t.floor_candidate, floor);
        }
        if (now - t.floor_since >= ui_adaptive::approach_dwell_ms) {
          t.floor_value = t.floor_confirmed ? std::max(t.floor_value, t.floor_candidate) : t.floor_candidate;
          t.floor_confirmed = true;
          t.floor_approaching = false;
        }
      } else {
        t.floor_approaching = false;
      }
    }

    std::array<track, max_tracks> tracks_ {};
    counters counters_;
    result last_;
    std::vector<part> last_parts_;
    std::map<std::uint64_t, bool> gates_;  // per route signature: open after a pass, closed after a fail
    log_depth previous_q_;
    std::vector<flow> previous_decoded_;
    std::uint64_t decoded_sequence_ {}, table_sequence_ {}, previous_index_ {}, clock_ {}, scope_ {}, winner_ {};
    bool have_decoded_ {}, table_fresh_ {}, have_clock_ {}, have_scope_ {}, have_winner_ {};
  };
}  // namespace sunshine_game3d::ui_anchor
