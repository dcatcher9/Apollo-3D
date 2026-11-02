// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// A synthetic 2.5-D scene for the world-anchored UI sequence test
// (test_game3d_ui_anchor_sequence.cpp): flat rectangles at fixed inverse
// depth over a far wall, moved each real frame by a camera whose image motion
// is a rotation term plus a translation term proportional to inverse depth
// (plus an optional forward term), and UI elements either attached to an
// owner's projected point or placed by a script. It renders what one real
// frame gives the anchor passes: the P1 weight mask and the displayed field
// at output resolution, inverse depth and motion vectors at render
// resolution, with exact motion and depth (no game's misdeclarations unless a
// test injects them).
#include "game3d_ui_anchor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace sunshine_game3d_test::anchor_scene {
  namespace anchor = sunshine_game3d::ui_anchor;

  // Image motion of one real frame, in output pixels: a point at inverse depth
  // rho and position p moves by rotation + translation * rho + forward * rho * (p - centre);
  // wide makes a yaw's image motion grow toward the frame's sides as a wide field of view does.
  struct camera {
    double rx {}, ry {}, tx {}, ty {}, forward {}, wide {};
  };

  struct layer {
    double x0 {}, y0 {}, x1 {}, y1 {};  // continuous output-pixel bounds
    double rho {};
    double vx {}, vy {};  // its own image motion per frame
    bool visible = true;
    bool wrap {};  // re-enter on the far side when it leaves the frame (background pillars)
    double ox {}, oy {};  // origin of its depth relief, moving with it
  };

  enum class label { hud, world, free };

  struct ui_element {
    std::string name;
    label kind = label::hud;
    double width {}, height {};
    int owner = -1;            // attached to this layer's top centre, else placed by script
    double dx {}, dy {};       // offset of its top-left from the owner's anchor
    unsigned lag {};           // frames behind its owner's projection
    std::function<std::pair<double, double>(unsigned frame)> place;  // top-left when not attached
    std::function<bool(unsigned frame)> shown;
    bool halo = true;
    bool fill_drain {};        // a drain inside an unchanged frame: inner fill varies
    // State.
    double x {}, y {};
    std::vector<std::pair<double, double>> history;  // owner anchors, newest last
  };

  struct scene {
    unsigned width = 640, height = 360, render_width = 320, render_height = 180;
    double wall_rho = .05;
    double gain = .03, zero = .4, bound = .02, plane = .005;  // h = clamp(gain * (rho - zero), -C, C)
    // Depth relief: each surface's inverse depth varies by this fraction in a
    // pattern that moves rigidly with it, as a real surface's shape does. The
    // displayed field uses the surface's base depth (conditioning smooths relief).
    double relief = .15;
    double wall_ox {}, wall_oy {};
    std::uint32_t ms_per_frame = 16;
    std::vector<layer> layers;
    std::vector<ui_element> ui;
    unsigned frame_number {};
    camera last;

    // Buffers of the current frame.
    std::vector<float> weight;
    std::vector<double> field;
    std::vector<float> rho;
    std::vector<std::array<float, 2>> motion;
    std::vector<int> owner_of;  // per output pixel: the UI element index whose texel it is, or -1

    [[nodiscard]] anchor::flow flow_at(double x, double y, double depth, const camera &c) const {
      const double side = (x - width / 2.) / (width / 2.);
      return {c.rx * (1 + c.wide * side * side) + c.tx * depth + c.forward * depth * (x - width / 2.),
        c.ry + c.ty * depth + c.forward * depth * (y - height / 2.)};
    }

    // Background pillars at the given inverse depths, spread over the frame.
    void add_pillars(const std::vector<double> &depths, double top, double bottom, double spacing, double pillar_width) {
      double x = 10;
      for (std::size_t i = 0; x < width + 4 * spacing; ++i, x += spacing) {
        const double depth = depths[i % depths.size()];
        layers.push_back({x, top + 7 * double(i % 3), x + pillar_width, bottom - 5 * double(i % 4), depth, 0, 0, true, true});
      }
    }

    // Moves every layer by one real frame of camera motion and its own motion,
    // then the UI.
    void advance(const camera &c) {
      last = c;
      ++frame_number;
      const double span = width + 8 * 60.;
      const auto w = flow_at(width / 2., height / 2., wall_rho, c);
      wall_ox += w.x;
      wall_oy += w.y;
      for (auto &l : layers) {
        const auto a = flow_at(l.x0, l.y0, l.rho, c), b = flow_at(l.x1, l.y1, l.rho, c);
        const auto m = flow_at((l.x0 + l.x1) / 2, (l.y0 + l.y1) / 2, l.rho, c);
        l.ox += m.x + l.vx;
        l.oy += m.y + l.vy;
        l.x0 += a.x + l.vx;
        l.y0 += a.y + l.vy;
        l.x1 += b.x + l.vx;
        l.y1 += b.y + l.vy;
        if (l.wrap) {
          if (l.x1 < -200) {
            l.x0 += span;
            l.x1 += span;
            l.ox += span;
          } else if (l.x0 > width + 200) {
            l.x0 -= span;
            l.x1 -= span;
            l.ox -= span;
          }
        }
      }
      place_ui();
    }

    void place_ui() {
      for (auto &e : ui) {
        if (e.owner >= 0) {
          const auto &o = layers[std::size_t(e.owner)];
          e.history.emplace_back((o.x0 + o.x1) / 2, o.y0);
          const auto &anchor = e.history.size() > e.lag ? e.history[e.history.size() - 1 - e.lag] : e.history.front();
          e.x = anchor.first + e.dx;
          e.y = anchor.second + e.dy;
          if (e.history.size() > 8) e.history.erase(e.history.begin());
        } else if (e.place) {
          const auto p = e.place(frame_number);
          e.x = p.first;
          e.y = p.second;
        }
      }
    }

    // A tabulated sine: the relief is evaluated at every render texel of every frame.
    [[nodiscard]] static double wave(double phase) {
      static const std::vector<double> table = [] {
        std::vector<double> t(4097);
        for (std::size_t i = 0; i < t.size(); ++i) t[i] = std::sin(2 * 3.14159265358979323846 * double(i) / 4096);
        return t;
      }();
      double turns = phase / (2 * 3.14159265358979323846);
      turns -= std::floor(turns);
      const double at = turns * 4096;
      const auto i = std::size_t(at);
      return table[i] + (table[std::min<std::size_t>(i + 1, 4096)] - table[i]) * (at - double(i));
    }

    [[nodiscard]] double relief_at(double x, double y) const {
      return 1 + relief * (.6 * wave(x * .21) * wave(y * .17) + .4 * wave(x * .093 + y * .131));
    }

    [[nodiscard]] double h_of(double depth) const {
      return std::clamp(gain * (depth - zero), -bound, bound);
    }

    // The front-most visible layer's index at a point, or -1 for the wall.
    [[nodiscard]] int front(double x, double y) const {
      int best = -1;
      double nearest = -1;
      for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto &l = layers[i];
        if (l.visible && x >= l.x0 && x < l.x1 && y >= l.y0 && y < l.y1 && l.rho > nearest) {
          nearest = l.rho;
          best = int(i);
        }
      }
      return best;
    }

    [[nodiscard]] static int texel(double v) {
      return int(std::lround(v));
    }

    // Integer bounds of a UI element's core this frame.
    [[nodiscard]] anchor::box core_of(const ui_element &e) const {
      return {texel(e.x), texel(e.y), texel(e.x + e.width) - 1, texel(e.y + e.height) - 1};
    }

    [[nodiscard]] bool showing(const ui_element &e) const {
      return !e.shown || e.shown(frame_number);
    }

    // Renders the frame's buffers with the given motion-vector scale.
    void render(double scale_x, double scale_y) {
      const std::size_t pixels = std::size_t(width) * height, texels = std::size_t(render_width) * render_height;
      weight.assign(pixels, 0.f);
      owner_of.assign(pixels, -1);
      field.assign(pixels, h_of(wall_rho));
      rho.assign(texels, float(wall_rho));
      motion.assign(texels, {0.f, 0.f});
      // Painter's order: far layers first.
      std::vector<std::size_t> order(layers.size());
      for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return layers[a].rho < layers[b].rho; });
      std::vector<int> front_render(texels, -1);
      for (const auto i : order) {
        const auto &l = layers[i];
        if (!l.visible) continue;
        const int x0 = std::max(0, int(std::ceil(l.x0 - .5))), x1 = std::min(int(width) - 1, int(std::ceil(l.x1 - .5)) - 1);
        const int y0 = std::max(0, int(std::ceil(l.y0 - .5))), y1 = std::min(int(height) - 1, int(std::ceil(l.y1 - .5)) - 1);
        for (int y = y0; y <= y1; ++y)
          for (int x = x0; x <= x1; ++x) field[std::size_t(y) * width + std::size_t(x)] = h_of(l.rho);
        // Render texel centres in output pixels: (i + 0.5) * W / W_mv in [x0, x1).
        const auto first = [](double edge, unsigned output, unsigned render) {
          return std::max(0, int(std::ceil(edge * render / output - .5)));
        };
        const int i0 = first(l.x0, width, render_width), i1 = std::min(int(render_width), first(l.x1, width, render_width));
        const int j0 = first(l.y0, height, render_height), j1 = std::min(int(render_height), first(l.y1, height, render_height));
        for (int j = j0; j < j1; ++j)
          for (int i2 = i0; i2 < i1; ++i2) front_render[std::size_t(j) * render_width + std::size_t(i2)] = int(i);
      }
      // Motion vectors: the motion that brought each texel's visible point here,
      // as a vector to its previous position: mv * scale = -flow / (W, H).
      for (unsigned j = 0; j < render_height; ++j) {
        for (unsigned i = 0; i < render_width; ++i) {
          const std::size_t at = std::size_t(j) * render_width + i;
          const double cx = (i + .5) * width / render_width, cy = (j + .5) * height / render_height;
          const int l = front_render[at];
          const double depth = l >= 0 ? layers[std::size_t(l)].rho : wall_rho;
          const double relief_depth = l >= 0 ? depth * relief_at(cx - layers[std::size_t(l)].ox, cy - layers[std::size_t(l)].oy) :
                                               depth * relief_at(cx - wall_ox, cy - wall_oy);
          // Each surface moves rigidly: its motion is its base depth's, relief included.
          auto f = flow_at(cx, cy, depth, last);
          if (l >= 0) {
            f.x += layers[std::size_t(l)].vx;
            f.y += layers[std::size_t(l)].vy;
          }
          rho[at] = float(relief_depth);
          motion[at] = {float(-f.x / (width * scale_x)), float(-f.y / (height * scale_y))};
        }
      }
      for (std::size_t k = 0; k < ui.size(); ++k) {
        const auto &e = ui[k];
        if (!showing(e)) continue;
        const auto b = core_of(e);
        for (int y = b.y0 - 1; y <= b.y1 + 1; ++y) {
          for (int x = b.x0 - 1; x <= b.x1 + 1; ++x) {
            if (x < 0 || y < 0 || x >= int(width) || y >= int(height)) continue;
            const bool core = b.contains(x, y);
            if (!core && !e.halo) continue;
            float w = core ? 1.f : .5f;
            if (core && e.fill_drain) {
              // The frame stays opaque; the fill inside is a softer weight that drains from the right.
              const bool rim = x == b.x0 || x == b.x1 || y == b.y0 || y == b.y1;
              const double fill = 1 - .5 * std::fmod(frame_number * .02, 1.);
              if (!rim && x > b.x0 + fill * (b.x1 - b.x0)) w = .5f;
            }
            const std::size_t at = std::size_t(y) * width + std::size_t(x);
            if (w > weight[at]) {
              weight[at] = w;
              owner_of[at] = int(k);
            }
          }
        }
      }
    }
  };
}  // namespace sunshine_game3d_test::anchor_scene
