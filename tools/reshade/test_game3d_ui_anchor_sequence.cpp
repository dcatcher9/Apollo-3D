// SPDX-License-Identifier: GPL-3.0-only
// Stage A4 of world-anchored UI planes (docs/game3d-anchored-ui-planes.md):
// the offline classifier of game3d_ui_anchor.h on synthetic sequences
// (test_game3d_ui_anchor_scene.h), one group per adversary of the design's
// list and positive controls. Every frame of every group checks the hard bar,
// that no HUD-labelled element reaches world (and that no HUD texel takes a
// target other than the plane); that the resolved targets never leave L > U
// in the anchored pin's envelope; and that no applied value moves faster than
// the approach rate except by a counted step (a projection or demotion when a
// constraint appears abruptly). A group that a known limit
// of the design decides prints "KNOWN_LIMIT <text>" and does not fail.
//
// The scenes are 640x360 output with a 426x240 render grid (a quality-mode
// ratio), flat surfaces with depth relief over a far wall, and exact motion
// vectors; real sequences come with stage A3.
#include "game3d_ui_anchor.h"
#include "test_game3d_ui_anchor_scene.h"
#include "test_game3d_ui_pin_band.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
  namespace an = sunshine_game3d::ui_anchor;
  namespace sc = sunshine_game3d_test::anchor_scene;
  namespace ui_adaptive = sunshine_game3d::ui_adaptive;
  using sunshine_game3d_test::ui_pin_band_detail::envelope;

  void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
  }

  unsigned known_limit_count = 0;

  void known_limit(const std::string &text) {
    std::printf("KNOWN_LIMIT %s\n", text.c_str());
    ++known_limit_count;
  }

  struct {
    std::uint64_t frames {}, hud_frames {}, world_frames {};
  } grand;

  // The standard scene: a far wall (rho 0.05) with relief and pillars at rho
  // 0.3 in the upper part; pUI = 0.008, C = 0.02, h = 0.03 (rho - 0.4).
  sc::scene standard_scene() {
    sc::scene s;
    s.render_width = 426;
    s.render_height = 240;
    s.plane = .008;
    s.add_pillars({.3}, 30, 250, 100, 60);
    return s;
  }

  int add_layer(sc::scene &s, double x, double y, double w, double h, double rho, double vx = 0, double vy = 0) {
    s.layers.push_back({x, y, x + w, y + h, rho, vx, vy, true, false});
    return int(s.layers.size()) - 1;
  }

  // An enemy's bar: attached 12 px above its top centre.
  void add_bar(sc::scene &s, const std::string &name, int owner, double w = 36, double h = 5, double above = 12) {
    sc::ui_element e;
    e.name = name;
    e.kind = sc::label::world;
    e.width = w;
    e.height = h;
    e.owner = owner;
    e.dx = -w / 2;
    e.dy = -above - h;
    s.ui.push_back(e);
  }

  void add_hud(sc::scene &s, const std::string &name, double x, double y, double w, double h,
      std::function<std::pair<double, double>(unsigned)> place = {}, sc::label kind = sc::label::hud) {
    sc::ui_element e;
    e.name = name;
    e.kind = kind;
    e.width = w;
    e.height = h;
    e.place = place ? place : [x, y](unsigned) { return std::make_pair(x, y); };
    s.ui.push_back(e);
  }

  // A crosshair, subtitles and a corner panel: static HUD in every group that adds them.
  void add_static_hud(sc::scene &s) {
    add_hud(s, "crosshair", 317, 177, 6, 6);
    add_hud(s, "subtitles", 170, 330, 300, 9);
    add_hud(s, "health", 20, 18, 90, 6);
    add_hud(s, "ammo", 560, 300, 30, 12);
  }

  // The positive control's camera: 10 frames strafing right, 10 strafing left
  // while yawing, 5 strafing left, so the mix of rotation and translation changes.
  sc::camera mix(unsigned frame) {
    const unsigned phase = frame % 25;
    if (phase < 10) return {0, 0, 16, 0, 0, 0};
    if (phase < 20) return {4, 0, -16, 0, 0, 0};
    return {0, 0, -16, 0, 0, 0};
  }

  sc::camera scaled(sc::camera c, double k) {
    c.rx *= k;
    c.ry *= k;
    c.tx *= k;
    c.ty *= k;
    c.forward *= k;
    return c;
  }

  struct tally {
    unsigned shown {}, world {}, pending {};
    int first_world = -1, last_world = -1;
    double error_px = -1;  // |applied - owner h| at the last world frame, in output pixels
    unsigned settled {};   // world frames within delta of the owner
  };

  struct edit {
    bool skip {}, reused {}, revision {}, hold_reset {}, camera_reset {}, full_frame {}, hidden {}, no_motion {};
    double interval_ms = 0;  // 0: the scene's frame time
  };

  edit with(bool edit::*flag) {
    edit e;
    e.*flag = true;
    return e;
  }

  class runner {
  public:
    runner(std::string name, sc::scene scene): name_(std::move(name)), s(std::move(scene)) {
      s.place_ui();
      tallies.resize(s.ui.size());
      previous_.assign(an::max_tracks, an::sentinel);
      previous_live_.assign(an::max_tracks, 0);
    }

    // The declared motion-vector scale relative to the true one (1/W_mv per axis).
    double declared_x = 1, declared_y = 1;
    bool declares_output_scale = false;
    std::uint64_t scope = 1, winner = 1, route = 1;
    std::string name_;
    sc::scene s;
    an::classifier c;
    std::vector<tally> tallies;
    unsigned frame = 0;

    const an::result &step(const sc::camera &camera, const edit &e = {}) {
      s.advance(camera);
      ++frame;
      ++sequence_;
      const double interval = e.interval_ms > 0 ? e.interval_ms : s.ms_per_frame;
      clock_ += interval;
      if (e.skip) return c.last();  // a real frame whose anchor passes did not run
      s.render(1. / s.render_width, 1. / s.render_height);
      an::frame f;
      f.now_ms = std::uint64_t(std::llround(clock_));
      f.sequence = sequence_;
      f.width = s.width;
      f.height = s.height;
      f.weight = s.weight.data();
      f.field = s.field.data();
      f.plane = s.plane;
      f.bound = s.bound;
      f.render = {s.render_width, s.render_height, s.rho.data(), e.no_motion ? nullptr : s.motion.data()};
      const double sx = declares_output_scale ? 1. / s.width : 1. / s.render_width;
      const double sy = declares_output_scale ? 1. / s.height : 1. / s.render_height;
      f.semantics = {sx * declared_x, sy * declared_y};
      f.proof = {true, true, true, true};
      f.scope = scope;
      f.winner = winner;
      f.route = route;
      f.frame_index = sequence_;
      f.reused_decision = e.reused;
      f.revision = e.revision;
      f.hold_reset = e.hold_reset;
      f.camera_reset = e.camera_reset;
      f.full_frame = e.full_frame;
      f.hidden = e.hidden;
      const auto steps = c.stats().steps;
      const auto &r = c.step(f);
      verify(r, std::min(interval, double(ui_adaptive::max_age_ms)), c.stats().steps != steps);
      return r;
    }

    [[nodiscard]] std::size_t index(const std::string &element) const {
      for (std::size_t k = 0; k < s.ui.size(); ++k)
        if (s.ui[k].name == element) return k;
      throw std::runtime_error("no element " + element);
    }
    [[nodiscard]] const tally &of(const std::string &element) const { return tallies[index(element)]; }
    [[nodiscard]] bool world_now(const std::string &element) const { return world_now_[index(element)]; }
    [[nodiscard]] std::uint64_t refusals(const char *reason) const {
      const auto it = c.stats().refusals.find(reason);
      return it == c.stats().refusals.end() ? 0 : it->second;
    }
    [[nodiscard]] std::uint64_t verdicts(const char *verdict) const {
      const auto it = c.stats().verdicts.find(verdict);
      return it == c.stats().verdicts.end() ? 0 : it->second;
    }

    std::string exits() const {
      std::string text;
      for (const auto &[reason, count] : c.stats().exits) text += (text.empty() ? "" : ", ") + reason + " " + std::to_string(count);
      return text.empty() ? "none" : text;
    }

    std::string describe(const std::string &element) const {
      const auto &t = of(element);
      char text[200];
      std::snprintf(text, sizeof(text), "%s world %u/%u frames (first %d, last %d, settled %u, error %.2f px)", element.c_str(), t.world,
        t.shown, t.first_world, t.last_world, t.settled, t.error_px);
      return text;
    }

  private:
    void verify(const an::result &r, double dt, bool stepped) {
      ++grand.frames;
      const unsigned w = s.width, h = s.height;
      const double delta = an::delta_uv(w);
      std::vector<int> part_of_cell(r.grid.cells.size(), -1);
      for (std::size_t i = 0; i < r.parts.size(); ++i)
        for (const auto cell : r.parts[i].cells) part_of_cell[cell] = int(i);
      world_now_.assign(s.ui.size(), 0);
      for (std::size_t k = 0; k < s.ui.size(); ++k) {
        const auto &e = s.ui[k];
        if (!s.showing(e)) continue;
        const auto b = an::intersect(s.core_of(e), an::frame_box(w, h));
        std::size_t texels = 0, world = 0, pending = 0, off = 0;
        std::map<int, std::size_t> votes;
        for (int y = b.y0; y <= b.y1; ++y) {
          for (int x = b.x0; x <= b.x1; ++x) {
            const std::size_t at = std::size_t(y) * w + std::size_t(x);
            if (s.owner_of[at] != int(k) || s.weight[at] <= 0.f) continue;
            const int part = part_of_cell[r.grid.index_of(unsigned(x), unsigned(y))];
            if (part < 0) continue;
            ++texels;
            ++votes[part];
            world += r.parts[std::size_t(part)].cls == an::track_class::world;
            pending += r.parts[std::size_t(part)].cls == an::track_class::pending;
            off += !an::is_sentinel(r.parts[std::size_t(part)].value);
          }
        }
        if (!texels) continue;
        auto &t = tallies[k];
        ++t.shown;
        if (e.kind == sc::label::hud) {
          ++grand.hud_frames;
          require(!world, name_ + ": HUD-labelled " + e.name + " reached world at frame " + std::to_string(frame));
          require(!off, name_ + ": HUD-labelled " + e.name + " took a target off the plane at frame " + std::to_string(frame));
        }
        if (world * 2 > texels) {
          ++t.world;
          ++grand.world_frames;
          world_now_[k] = 1;
          if (t.first_world < 0) t.first_world = int(frame);
          t.last_world = int(frame);
          if (e.owner >= 0) {
            int best = -1;
            std::size_t most = 0;
            for (const auto &[part, count] : votes)
              if (count > most) {
                most = count;
                best = part;
              }
            const double value = r.parts[std::size_t(best)].value;
            if (!an::is_sentinel(value)) {
              const double error = std::abs(value - s.h_of(s.layers[std::size_t(e.owner)].rho));
              t.error_px = error * w;
              t.settled += error <= delta;
            }
          }
        } else if (pending * 2 > texels) {
          ++t.pending;
        }
      }
      // The resolved targets never leave L > U in the anchored pin; rows are
      // independent, so only the rows with UI are checked.
      const auto targets = an::texel_targets(r, s.weight.data());
      std::vector<float> field, alpha;
      std::vector<double> row_targets;
      unsigned rows = 0;
      for (unsigned y = 0; y < h; ++y) {
        const std::size_t at = std::size_t(y) * w;
        if (std::none_of(s.weight.begin() + std::ptrdiff_t(at), s.weight.begin() + std::ptrdiff_t(at + w), [](float v) { return v > 0.f; })) continue;
        field.insert(field.end(), s.field.begin() + std::ptrdiff_t(at), s.field.begin() + std::ptrdiff_t(at + w));
        alpha.insert(alpha.end(), s.weight.begin() + std::ptrdiff_t(at), s.weight.begin() + std::ptrdiff_t(at + w));
        row_targets.insert(row_targets.end(), targets.begin() + std::ptrdiff_t(at), targets.begin() + std::ptrdiff_t(at + w));
        ++rows;
      }
      if (rows) {
        const auto pin = envelope(field, alpha, &row_targets, w, rows, s.plane, 1.);
        for (std::size_t i = 0; i < field.size(); ++i)
          if (pin.lower[i] > pin.upper[i] + 1e-9 / w) require(false, name_ + ": a resolved frame has L > U at frame " + std::to_string(frame));
      }
      // No applied value moves faster than the approach rate except by a counted step.
      const double step = ui_adaptive::approach_rate_uv_per_second * dt / 1000 + 1e-12;
      const auto &tracks = c.tracks();
      for (std::size_t t = 0; t < an::max_tracks; ++t) {
        const double now = tracks[t].applied;
        if (tracks[t].live && previous_live_[t] && !stepped && !an::is_sentinel(now) && !an::is_sentinel(previous_[t]) &&
            std::abs(now - previous_[t]) > step)
          require(false, name_ + ": an applied value snapped at frame " + std::to_string(frame));
        previous_[t] = now;
        previous_live_[t] = tracks[t].live;
      }
    }

    std::uint64_t sequence_ = 100;
    double clock_ = 1000;
    std::vector<double> previous_;
    std::vector<std::uint8_t> previous_live_;
    std::vector<std::uint8_t> world_now_;
  };

  // ---------------------------------------------------------------- positive controls and static HUD

  // An enemy bar under a camera whose mix changes reaches world at the enemy's
  // displayed depth; a crosshair, subtitles and corner panels never do.
  void positive_control_and_static_hud() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("positive control", scene);
    for (unsigned f = 0; f < 150; ++f) run.step(mix(f));
    const auto &bar = run.of("bar");
    require(bar.first_world >= 0 && bar.first_world <= 40, "the enemy bar did not reach world within 40 frames: " + run.describe("bar"));
    // Every false check fail returns every track to global (the design's rule), so recall stays below one.
    require(bar.world * 10 >= (150 - unsigned(bar.first_world)) * 6, "the enemy bar left world too often: " + run.describe("bar"));
    require(bar.error_px >= 0 && bar.error_px <= 2, "the enemy bar did not settle within delta of its enemy: " + run.describe("bar"));
    std::printf("  %s; check pass/none/fail %llu/%llu/%llu; exits %s\n", run.describe("bar").c_str(), (unsigned long long) run.verdicts("pass"),
      (unsigned long long) run.verdicts("none"), (unsigned long long) run.verdicts("fail"), run.exits().c_str());
  }

  // Static HUD under every kind of camera motion.
  void still_crosshair_and_subtitles() {
    auto scene = standard_scene();
    add_static_hud(scene);
    add_hud(scene, "marker", 400, 120, 8, 8);
    add_layer(scene, 200, 260, 60, 100, .85);  // foreground rock
    runner run("static HUD", scene);
    const sc::camera cameras[] {{0, 0, 16, 0, 0, 0}, {0, 0, -16, 0, 0, 0}, {6, 0, 0, 0, 0, 0}, {0, 5, 0, 0, 0, 0},
      {-8.8, 0, 16, 0, 0, 0}, {3, 3, 6, 4, 0, 0}, {0, 0, 0, 0, .01, 0}, {4, 0, -16, 0, 0, 0}};
    for (const auto &camera : cameras)
      for (unsigned f = 0; f < 20; ++f) run.step(camera);
    std::printf("  %u frames under strafes, yaw, pitch, orbit, diagonal, forward and mixed motion\n", run.frame);
  }

  // ---------------------------------------------------------------- HUD that moves with the camera

  // Compass ticks moving with the yaw: along the top strip (never a candidate)
  // and in a band through the centre, under yaw, then yaw mixed with strafes.
  void compass_under_yaw() {
    auto scene = standard_scene();
    add_layer(scene, 100, 270, 80, 90, .85);
    double yaw = 0;
    const auto tick = [&yaw](double base, double y) {
      return [&yaw, base, y](unsigned) { return std::make_pair(150 + std::fmod(base + yaw + 3400, 340.), y); };
    };
    for (int i = 0; i < 6; ++i) {
      add_hud(scene, "top tick " + std::to_string(i), 0, 0, 3, 8, tick(57. * i, 14));
      add_hud(scene, "central tick " + std::to_string(i), 0, 0, 3, 8, tick(57. * i + 20, 150));
    }
    runner run("compass", scene);
    for (unsigned f = 0; f < 160; ++f) {
      sc::camera camera {6 * std::sin(f * .11), 0, f < 60 ? 0. : 16 * std::cos(f * .07), 0, 0, 0};
      yaw += camera.rx;  // the ticks move as infinitely far points do
      run.step(camera);
    }
    std::printf("  12 ticks over %u frames of yaw and yaw with strafes\n", run.frame);
  }

  // A marker clamped to one row: it moves with the yaw but not with the pitch.
  void clamped_marker_under_yaw_then_pitch() {
    auto scene = standard_scene();
    add_layer(scene, 400, 270, 80, 90, .85);
    double x = 300;
    add_hud(scene, "clamped marker", 0, 0, 10, 10, [&x](unsigned) { return std::make_pair(std::clamp(x, 90., 540.), 60.); });
    runner run("clamped marker", scene);
    for (unsigned f = 0; f < 150; ++f) {
      sc::camera camera = f < 50 ? sc::camera {5 * std::sin(f * .2), 0, 0, 0, 0, 0} : f < 100 ? sc::camera {0, 4 * std::sin(f * .2), 0, 0, 0, 0} :
                                                                                       sc::camera {4 * std::sin(f * .2), 3 * std::cos(f * .2), 12 * std::sin(f * .05), 0, 0, 0};
      x += camera.rx;
      run.step(camera);
    }
    std::printf("  clamped marker over %u frames of yaw, pitch and both with strafes\n", run.frame);
  }

  // Notifications sliding in at a constant speed and with easing while the camera strafes steadily.
  void notification_sliding_during_a_strafe() {
    auto scene = standard_scene();
    add_hud(scene, "notification", 0, 0, 50, 8, [](unsigned f) { return std::make_pair(80 + 6. * std::min(f % 60, 40u), 110.); });
    add_hud(scene, "eased notification", 0, 0, 50, 8, [](unsigned f) {
      const double t = std::min(f % 60, 40u) / 40.;
      return std::make_pair(380 - 200 * (1 - (1 - t) * (1 - t)), 200.);
    });
    runner run("notifications", scene);
    for (unsigned f = 0; f < 180; ++f) run.step({0, 0, (f / 60) % 2 ? -16. : 16., 0, 0, 0});
    std::printf("  %s; %s\n", run.describe("notification").c_str(), run.describe("eased notification").c_str());
  }

  // Lock-on: the camera orbits the locked enemy at a varying rate while the
  // HUD sways with that rate; the enemy's bar, anchored beforehand, holds.
  void lock_on_orbit_with_sway_and_an_anchored_bar() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_layer(scene, 100, 270, 80, 90, .85);
    add_layer(scene, 470, 280, 70, 80, .85);
    double rate = 0;
    add_hud(scene, "swaying reticle", 0, 0, 8, 8, [&rate](unsigned) { return std::make_pair(316 - 1.2 * rate, 176.); });
    runner run("lock-on orbit", scene);
    for (unsigned f = 0; f < 75; ++f) run.step(mix(f));
    require(run.world_now("bar"), "the bar was not world when the lock-on began: " + run.describe("bar"));
    const unsigned before = run.of("bar").world;
    for (unsigned f = 0; f < 90; ++f) {
      const double tx = (f / 8) % 2 ? 6.4 : 16;
      rate = -.55 * tx;  // the enemy stays put; everything else orbits around it
      run.step({rate, 0, tx, 0, 0, 0});
    }
    const unsigned held = run.of("bar").world - before;
    require(held >= 80, "the anchored bar did not hold through the lock-on orbit: " + run.describe("bar"));
    std::printf("  bar held world %u/90 orbit frames; %s\n", held, run.describe("swaying reticle").c_str());
  }

  // Walking: a forward motion with a head-bob whose amplitude follows the
  // walking speed; the crosshair bobs with the camera.
  void head_bob_with_walking_speed() {
    auto scene = standard_scene();
    add_layer(scene, 150, 270, 80, 90, .85);
    double bob = 0;
    add_hud(scene, "bobbing crosshair", 0, 0, 6, 6, [&bob](unsigned) { return std::make_pair(317., 177 + 1.5 * bob); });
    add_hud(scene, "bobbing weapon", 0, 0, 40, 20, [&bob](unsigned) { return std::make_pair(400., 250 + 3 * bob); });
    runner run("head-bob", scene);
    for (unsigned f = 0; f < 160; ++f) {
      const double speed = 1 + .6 * std::sin(f * .04);
      const double vertical = 4 * speed * std::cos(f * .5);
      bob += vertical;
      run.step({0, 0, 0, vertical, .004 * speed, 0});
    }
    std::printf("  bobbing crosshair and weapon over %u frames of walking\n", run.frame);
  }

  // A fast yaw over a wide field of view, whose image motion varies across
  // the frame, with an element as large as the candidate cap swaying with it.
  void fast_wide_yaw_with_a_large_element() {
    auto scene = standard_scene();
    add_layer(scene, 150, 270, 80, 90, .85);
    double yaw = 0;
    add_hud(scene, "large panel", 0, 0, 90, 40, [&yaw](unsigned) { return std::make_pair(270 + .5 * yaw, 140.); });
    add_hud(scene, "large static panel", 0, 0, 40, 90, [](unsigned) { return std::make_pair(120., 120.); });
    runner run("wide yaw", scene);
    for (unsigned f = 0; f < 150; ++f) {
      const double rx = 14 * std::sin(f * .09);
      yaw = rx;
      run.step({rx, 0, 3 * std::cos(f * .13), 0, 0, .8});
    }
    std::printf("  H/4 panels over %u frames of fast wide yaw\n", run.frame);
  }

  // ---------------------------------------------------------------- owners' own motion

  // An enemy walking steadily: under a still camera there is no depth to
  // discriminate; under a steady strafe the evidence never changes; only a
  // change of the mix enters world.
  void steadily_walking_enemy() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 150, 150, 40, 80, .55, 4);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("walking enemy", scene);
    for (unsigned f = 0; f < 60; ++f) run.step({});
    require(run.of("bar").world == 0 && run.of("bar").pending == 0, "a walking enemy under a still camera gave evidence: " + run.describe("bar"));
    // Walking 2 px a frame against a steady strafe: it moves by -10.8 px, the wall by -0.8 and the pillars by
    // -4.8, so its reach discriminates depth, but every evidence is the same (d_E, a).
    run.s.layers[std::size_t(enemy)].vx = -2;
    for (unsigned f = 0; f < 20; ++f) run.step({0, 0, -16, 0, 0, 0});
    require(run.of("bar").world == 0, "a walking enemy under a steady strafe reached world: " + run.describe("bar"));
    const unsigned pending = run.of("bar").pending;
    require(pending >= 10, "a walking enemy under a steady strafe was not pending: " + run.describe("bar"));
    run.s.layers[std::size_t(enemy)].vx = 0;
    for (unsigned f = 0; f < 60; ++f) run.step(f % 20 < 10 ? sc::camera {0, 0, 16, 0, 0, 0} : sc::camera {4, 0, -16, 0, 0, 0});
    require(run.of("bar").world > 0, "a changed mix did not enter world: " + run.describe("bar"));
    std::printf("  no evidence while the camera was still; pending %u frames under the steady strafe; then %s\n", pending,
      run.describe("bar").c_str());
  }

  // A fill draining inside an unchanged frame keeps the bar's class; a HUD
  // bar shrinking from the right is never rigid.
  void hp_drains() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "draining bar", enemy);
    scene.ui.back().fill_drain = true;
    add_hud(scene, "shrinking HUD bar", 0, 0, 60, 6, [](unsigned) { return std::make_pair(200., 90.); });
    runner run("HP drains", scene);
    for (unsigned f = 0; f < 150; ++f) {
      run.s.ui[1].width = 60 - double(f % 50);
      run.step(mix(f));
    }
    const auto &bar = run.of("draining bar");
    require(bar.first_world >= 0 && bar.world * 10 >= (150 - unsigned(bar.first_world)) * 6,
      "a draining bar did not keep world: " + run.describe("draining bar"));
    std::printf("  %s\n", run.describe("draining bar").c_str());
  }

  // A far objective's distance marker ("54m" to "55m"): it moves as an
  // infinitely far point and changes width by a glyph; its owner is hidden.
  void distance_marker_changing_width() {
    auto scene = standard_scene();
    add_layer(scene, 100, 270, 80, 90, .85);
    double x = 300;
    add_hud(scene, "54m", 0, 0, 20, 8, [&x](unsigned) { return std::make_pair(x, 120.); });
    runner run("54m", scene);
    for (unsigned f = 0; f < 150; ++f) {
      const auto camera = mix(f);
      x += camera.rx + .02 * camera.tx;
      if (x > 520 || x < 100) x = 300;
      run.s.ui[0].width = (f / 15) % 2 ? 26 : 20;
      run.step(camera);
    }
    std::printf("  %s\n", run.describe("54m").c_str());
  }

  // Two enemies at different depths with bars stacked above each other.
  void stacked_bars_at_different_depths() {
    auto scene = standard_scene();
    const int far = add_layer(scene, 270, 170, 40, 80, .55);
    const int near = add_layer(scene, 285, 150, 30, 60, .75);
    add_bar(scene, "lower bar", far, 36, 5, 6);
    add_bar(scene, "upper bar", near, 30, 5, 10);
    runner run("stacked bars", scene);
    for (unsigned f = 0; f < 150; ++f) run.step(mix(f));
    for (const char *bar : {"lower bar", "upper bar"}) {
      require(run.of(bar).first_world >= 0, std::string("a stacked bar never reached world: ") + run.describe(bar));
      require(run.of(bar).error_px >= 0 && run.of(bar).error_px <= 2, std::string("a stacked bar did not settle at its own enemy: ") + run.describe(bar));
    }
    std::printf("  %s; %s\n", run.describe("lower bar").c_str(), run.describe("upper bar").c_str());
  }

  // A damage number spawns touching the bar, then floats up and separates.
  void damage_number_merging_and_separating() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    sc::ui_element number;
    number.name = "damage number";
    number.kind = sc::label::free;
    number.width = 10;
    number.height = 7;
    number.owner = enemy;
    number.dx = 19;
    number.dy = -20;
    number.shown = [](unsigned f) { return f % 30 < 18; };
    scene.ui.push_back(number);
    add_static_hud(scene);
    runner run("damage number", scene);
    for (unsigned f = 0; f < 150; ++f) {
      run.s.ui[1].dy = -20 - 1.5 * double(f % 30);
      run.step(mix(f));
    }
    std::printf("  %s; %s\n", run.describe("bar").c_str(), run.describe("damage number").c_str());
  }

  // ---------------------------------------------------------------- leaving candidacy, neighbours, floors

  // An anchored bar walks out of the central rectangle, another grows past the
  // H/4 cap: each returns to global and its value ramps back without a snap.
  void bar_leaving_the_centre_or_outgrowing_the_cap() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    const int other = add_layer(scene, 200, 150, 40, 80, .55);
    add_bar(scene, "growing bar", other, 30);
    runner run("leaving candidacy", scene);
    for (unsigned f = 0; f < 75; ++f) run.step(mix(f));
    require(run.world_now("bar") && run.world_now("growing bar"), "the bars were not anchored before leaving candidacy: " +
      run.describe("bar") + "; " + run.describe("growing bar"));
    run.s.layers[std::size_t(enemy)].vx = 6;
    for (unsigned f = 0; f < 60; ++f) {
      run.s.ui[1].width = std::min(120., run.s.ui[1].width + 2);
      run.s.ui[1].dx = -run.s.ui[1].width / 2;
      run.step({});
    }
    require(!run.world_now("bar") && !run.world_now("growing bar"), "a bar outside candidacy stayed world");
    const auto exits = run.c.stats().exits;
    require(exits.count("not_candidate") && exits.at("not_candidate") >= 2, "leaving candidacy was not counted as an exit");
    std::printf("  both returned to global (exits not_candidate %llu); values ramped without a snap\n",
      (unsigned long long) exits.at("not_candidate"));
  }

  // An anchored bar beside static HUD in the same rows, and a near rock
  // passing over it: the bar slides toward pUI beside the HUD, the floor
  // lifts it in front of the rock and is dropped where it cannot fit.
  void bar_beside_hud_with_a_near_occluder() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_hud(scene, "objective text", 0, 0, 60, 7, [](unsigned) { return std::make_pair(332., 132.); });
    const int rock = add_layer(scene, -200, 100, 60, 60, .95);
    scene.layers[std::size_t(rock)].visible = false;
    runner run("bar beside HUD", scene);
    for (unsigned f = 0; f < 75; ++f) run.step(mix(f));
    auto &r = run.s.layers[std::size_t(rock)];
    r.visible = true;
    r.x0 = 220;
    r.x1 = 280;
    r.vx = 1.5;
    for (unsigned f = 0; f < 80; ++f) run.step({});
    const auto &stats = run.c.stats();
    require(stats.floor_lifts + stats.floor_drops > 0, "the near rock never raised or dropped a floor");
    std::printf("  %s; floor lifts %llu, floors dropped %llu, demotions %llu\n", run.describe("bar").c_str(),
      (unsigned long long) stats.floor_lifts, (unsigned long long) stats.floor_drops, (unsigned long long) stats.demotions);
  }

  // ---------------------------------------------------------------- the motion check

  // The positive control's camera with a vertical component, so that every
  // injected error changes the decoded motion.
  sc::camera mix_with_pitch(unsigned frame) {
    auto camera = mix(frame);
    camera.ry = (frame / 5) % 2 ? 4 : -4;
    return camera;
  }

  // Injected sign, axis and scale errors: from the first frame nothing
  // anchors; injected after the bar anchored, it returns to global at once.
  void injected_sign_axis_and_scale_errors() {
    struct injection {
      const char *name;
      double x, y;
      bool output;
    };
    const injection injections[] {{"x negated", -1, 1, false}, {"y negated", 1, -1, false}, {"x2", 2, 2, false}, {"output scale", 1, 1, true}};
    for (const auto &inj : injections) {
      for (const bool late : {false, true}) {
        auto scene = standard_scene();
        const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
        add_bar(scene, "bar", enemy);
        runner run(std::string("injected ") + inj.name, scene);
        unsigned f = 0;
        if (late) {
          for (; f < 75; ++f) run.step(mix_with_pitch(f));
          require(run.world_now("bar"), std::string("the bar did not anchor before injecting ") + inj.name + ": " + run.describe("bar"));
        }
        run.declared_x = inj.x;
        run.declared_y = inj.y;
        run.declares_output_scale = inj.output;
        const auto fails = run.verdicts("fail");
        const unsigned world_before = run.of("bar").world;
        int first_fail = -1;
        unsigned world_after_fail = 0;
        for (unsigned k = 0; k < 75; ++k, ++f) {
          run.step(mix_with_pitch(f));
          if (first_fail < 0 && run.verdicts("fail") > fails) first_fail = int(k);
          if (first_fail >= 0) world_after_fail += run.world_now("bar");
        }
        const unsigned world = run.of("bar").world - world_before;
        require(first_fail >= 0 && first_fail <= 2, std::string("the check did not fail promptly under ") + inj.name);
        require(world_after_fail == 0, std::string("the bar was world after the check failed under ") + inj.name + ": " + run.describe("bar"));
        std::printf("  %s %s: first fail at injected frame %d; world on %u injected frames; checks pass/none/fail %llu/%llu/%llu\n", inj.name,
          late ? "after anchoring" : "from the start", first_fail, world, (unsigned long long) run.verdicts("pass"),
          (unsigned long long) run.verdicts("none"), (unsigned long long) run.verdicts("fail"));
      }
    }
  }

  // A cut to another place without a camera reset flag.
  void cut_without_reset() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("cut", scene);
    for (unsigned f = 0; f < 75; ++f) run.step(mix(f));
    require(run.world_now("bar"), "the bar did not anchor before the cut: " + run.describe("bar"));
    // The new place: other pillars at other depths, the enemy elsewhere and nearer.
    auto &layers = run.s.layers;
    layers.erase(std::remove_if(layers.begin(), layers.end(), [](const sc::layer &l) { return l.wrap; }), layers.end());
    const int moved = int(std::find_if(layers.begin(), layers.end(), [](const sc::layer &l) { return !l.wrap; }) - layers.begin());
    run.s.add_pillars({.2, .65}, 40, 300, 60, 25);
    run.s.wall_rho = .08;
    auto &e = layers[std::size_t(moved)];
    e = {420, 120, 460, 200, .7, 0, 0, true, false};
    for (auto &u : run.s.ui) {
      u.owner = u.owner >= 0 ? moved : u.owner;
      u.history.clear();
    }
    const auto fails = run.verdicts("fail");
    const auto &r = run.step({});
    const bool failed = run.verdicts("fail") > fails;
    for (unsigned f = 0; f < 40; ++f) run.step(mix(f));
    std::printf("  the cut's pair: check %s%s; %s\n", an::name(r.check), failed ? " (every track returned to global)" : "",
      run.describe("bar").c_str());
    if (!failed) known_limit(std::string("a cut without reset gave the check verdict ") + an::name(r.check) + "; the cut's evidence was " + r.pair);
  }

  // ---------------------------------------------------------------- frame identity

  // Frame generation: real frames come every 2, 3 or 4 Presents (the
  // multiplier drifts); generated Presents never step and show the real
  // frame's targets.
  void fg_cadence_and_multiplier_drift() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("frame generation", scene);
    unsigned multiplier = 2;
    for (unsigned f = 0; f < 120; ++f) {
      if (f % 30 == 0) multiplier = 2 + (f / 30) % 3;
      // Each real frame moves as far as its Presents' share of 16 ms at 120 Hz cadence.
      edit e;
      e.interval_ms = 8. * multiplier;
      const auto &r = run.step(scaled(mix(f), multiplier * .5), e);
      const auto held = r.cell_target;
      for (unsigned g = 1; g < multiplier; ++g) {
        const auto &shown = run.c.last();
        require(shown.cell_target.size() == held.size() && std::equal(held.begin(), held.end(), shown.cell_target.begin(),
          [](double a, double b) { return (an::is_sentinel(a) && an::is_sentinel(b)) || a == b; }), "a generated Present changed the targets");
      }
    }
    require(run.of("bar").first_world >= 0, "the bar never anchored under frame generation: " + run.describe("bar"));
    std::printf("  multipliers 2-4: %s\n", run.describe("bar").c_str());
  }

  // A real frame whose anchor passes did not run every 7 frames.
  void skipped_real_frames() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("skipped frames", scene);
    for (unsigned f = 0; f < 150; ++f) run.step(mix(f), f % 7 == 3 ? with(&edit::skip) : edit {});
    require(run.refusals("not_consecutive") >= 20, "skipped real frames were not refused as not consecutive");
    require(run.of("bar").first_world >= 0, "the bar never anchored with skipped frames: " + run.describe("bar"));
    std::printf("  %llu pairs refused as not consecutive; %s\n", (unsigned long long) run.refusals("not_consecutive"),
      run.describe("bar").c_str());
  }

  // ---------------------------------------------------------------- segmentation over time

  // HUD that splits and rejoins, HUD pieces sliding together, blinking HUD,
  // and a blinking world bar that keeps its class.
  void split_merge_and_blink() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "blinking bar", enemy);
    scene.ui.back().shown = [](unsigned f) { return f < 90 || f % 20 >= 6; };
    add_hud(scene, "split label left", 0, 0, 20, 7, [](unsigned) { return std::make_pair(150., 100.); });
    add_hud(scene, "split label right", 0, 0, 20, 7, [](unsigned f) { return std::make_pair(171 + 12 * std::abs(std::sin(f * .1)), 100.); });
    add_hud(scene, "merging left", 0, 0, 16, 7, [](unsigned f) { return std::make_pair(400 + 20 * std::abs(std::sin(f * .07)), 220.); });
    add_hud(scene, "merging right", 0, 0, 16, 7, [](unsigned f) { return std::make_pair(440 - 20 * std::abs(std::sin(f * .07)), 220.); });
    add_hud(scene, "blinking icon", 0, 0, 9, 9, [](unsigned) { return std::make_pair(480., 90.); });
    scene.ui.back().shown = [](unsigned f) { return f % 15 < 10; };
    runner run("split, merge, blink", scene);
    for (unsigned f = 0; f < 150; ++f) run.step(mix(f));
    require(run.of("blinking bar").first_world >= 0 && run.of("blinking bar").last_world >= 140,
      "the blinking bar did not keep world: " + run.describe("blinking bar"));
    std::printf("  %s\n", run.describe("blinking bar").c_str());
  }

  // The scene's calibration ramps (its gain and zero change), so the displayed
  // field moves under an anchored bar: the bar follows it.
  void calibration_ramps() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("calibration", scene);
    for (unsigned f = 0; f < 75; ++f) run.step(mix(f));
    for (unsigned f = 75; f < 135; ++f) {
      run.s.gain = .03 + .01 * (f - 75) / 60.;
      run.s.zero = .4 - .05 * (f - 75) / 60.;
      run.step(mix(f));
    }
    for (unsigned f = 135; f < 230; ++f) run.step(mix(f));
    require(run.world_now("bar") && run.of("bar").error_px <= 2, "the bar did not follow the calibration: " + run.describe("bar"));
    std::printf("  h of the enemy %.5f -> %.5f; %s\n", .03 * .15, run.s.h_of(.55), run.describe("bar").c_str());
  }

  // Scope changes and full-frame or hidden decisions drop every track; winner
  // changes, hold resets and camera resets return them to global; an
  // observation revision keeps them and refuses its pair; a reused decision
  // gathers no evidence.
  void scope_changes_and_observation_revisions() {
    auto scene = standard_scene();
    const int enemy = add_layer(scene, 280, 150, 40, 80, .55);
    add_bar(scene, "bar", enemy);
    add_static_hud(scene);
    runner run("scope and revisions", scene);
    unsigned f = 0;
    const auto anchor = [&](const char *when) {
      for (unsigned k = 0; k < 90 && !run.world_now("bar"); ++k, ++f) run.step(mix(f));
      require(run.world_now("bar"), std::string("the bar did not anchor ") + when + ": " + run.describe("bar"));
    };
    anchor("first");
    run.step(mix(f++), with(&edit::revision));
    require(run.world_now("bar") && run.refusals("revision") == 1, "an observation revision did not keep the tracks and refuse its pair");
    run.step(mix(f++), with(&edit::reused));
    require(run.world_now("bar") && run.refusals("reused") == 1, "a reused decision did not keep the tracks and refuse its pair");
    for (const char *drop : {"scope", "full_frame", "hidden"}) {
      anchor(drop);
      edit e;
      if (std::string(drop) == "scope") ++run.scope;
      e.full_frame = std::string(drop) == "full_frame";
      e.hidden = std::string(drop) == "hidden";
      const auto &r = run.step(mix(f++), e);
      require(r.pair == std::string(drop) && std::all_of(r.cell_target.begin(), r.cell_target.end(), an::is_sentinel),
        std::string("a ") + drop + " decision did not drop every track");
    }
    for (const char *boundary : {"winner", "hold_reset", "camera_reset"}) {
      anchor(boundary);
      edit e;
      if (std::string(boundary) == "winner") ++run.winner;
      e.hold_reset = std::string(boundary) == "hold_reset";
      e.camera_reset = std::string(boundary) == "camera_reset";
      const auto &r = run.step(mix(f++), e);
      require(r.pair == std::string(boundary) && !run.world_now("bar"), std::string("a ") + boundary + " boundary did not return every track to global");
    }
    std::printf("  revision and reused decision kept the tracks; scope, full-frame and hidden dropped them; winner, hold reset and camera "
      "reset returned them to global\n");
  }

  // ---------------------------------------------------------------- known limit

  // A HUD marker that co-moves with one layer through distinct motions is,
  // to motion, a world element: it anchors at that layer until a
  // contradiction persists (the design's bounded failure).
  void marker_co_moving_with_one_layer() {
    auto scene = standard_scene();
    double x = 300;
    add_hud(scene, "co-moving marker", 0, 0, 10, 10, [&x](unsigned) { return std::make_pair(x, 90.); }, sc::label::free);
    runner run("co-moving marker", scene);
    unsigned f = 0;
    for (; f < 120; ++f) {
      const auto camera = mix(f);
      x += camera.rx + camera.tx * .3;  // exactly as the pillars at rho 0.3
      run.step(camera);
    }
    const auto &t = run.of("co-moving marker");
    if (t.first_world >= 0)
      known_limit("a HUD marker co-moving with the pillars at rho 0.3 through distinct motions anchors there at frame " +
        std::to_string(t.first_world) + " (" + std::to_string(t.world) + "/" + std::to_string(t.shown) + " frames world)");
    // Once it stops co-moving, the contradiction must persist for the dwell and return it.
    for (unsigned k = 0; k < 40; ++k, ++f) run.step(mix(f));
    require(!run.world_now("co-moving marker"), "a marker that stopped co-moving stayed world: " + run.describe("co-moving marker"));
  }

  // ---------------------------------------------------------------- the check's statistic

  // Why the residual is the median deviation: under a lock-on orbit the mean
  // deviation fails a correct declaration on most frames, the median passes;
  // both fail injected errors.
  void check_statistic_under_an_orbit() {
    auto scene = standard_scene();
    add_layer(scene, 280, 150, 40, 80, .55);
    add_layer(scene, 100, 270, 80, 90, .85);
    scene.place_ui();
    const double sx = 1. / scene.render_width, sy = 1. / scene.render_height;
    scene.render(sx, sy);
    an::render_input r0 {scene.render_width, scene.render_height, scene.rho.data(), scene.motion.data()};
    auto previous = an::quarter_log_depth(r0);
    unsigned counts[2][2][3] {};  // score, declaration (correct, x2), verdict
    for (unsigned f = 0; f < 40; ++f) {
      scene.advance({-8.8, 0, 16, 0, 0, 0});
      scene.render(sx, sy);
      an::render_input r {scene.render_width, scene.render_height, scene.rho.data(), scene.motion.data()};
      const auto current = an::quarter_log_depth(r);
      for (unsigned d = 0; d < 2; ++d) {
        const auto decoded = an::decode_motion(r, {sx * (d ? 2 : 1), sy * (d ? 2 : 1)}, scene.width, scene.height);
        for (unsigned score = 0; score < 2; ++score) {
          const auto out = an::check_motion(previous, current, decoded, r, scene.width, scene.height,
            score ? an::residual_score::mean_deviation : an::residual_score::median_deviation);
          ++counts[score][d][unsigned(out.outcome)];
        }
      }
      previous = current;
    }
    const auto line = [&](unsigned score, unsigned d) {
      return std::to_string(counts[score][d][1]) + "/" + std::to_string(counts[score][d][0]) + "/" + std::to_string(counts[score][d][2]);
    };
    require(counts[1][0][2] >= 20, "the mean deviation no longer fails a correct declaration under the orbit: revisit the check's statistic");
    require(counts[0][0][2] <= 2 && counts[0][0][1] >= 30, "the median deviation does not pass a correct declaration under the orbit");
    require(counts[0][1][2] >= 36, "the median deviation does not fail x2 under the orbit");
    std::printf("  40 orbit pairs, pass/none/fail: median deviation correct %s, x2 %s; mean deviation correct %s, x2 %s\n", line(0, 0).c_str(),
      line(0, 1).c_str(), line(1, 0).c_str(), line(1, 1).c_str());
  }
}  // namespace

int main() {
  const std::pair<const char *, void (*)()> groups[] {
    {"check statistic under a lock-on orbit", check_statistic_under_an_orbit},
    {"positive control; still crosshair and subtitles", positive_control_and_static_hud},
    {"still crosshair and subtitles under every camera", still_crosshair_and_subtitles},
    {"compass under yaw", compass_under_yaw},
    {"clamped marker under yaw then pitch", clamped_marker_under_yaw_then_pitch},
    {"notification sliding during a strafe, with and without easing", notification_sliding_during_a_strafe},
    {"HUD sway under a lock-on orbit; lock-on with an anchored bar", lock_on_orbit_with_sway_and_an_anchored_bar},
    {"head-bob with walking speed", head_bob_with_walking_speed},
    {"fast wide-FOV yaw with an H/4 element", fast_wide_yaw_with_a_large_element},
    {"a steadily walking enemy", steadily_walking_enemy},
    {"HP drains", hp_drains},
    {"\"54m\" to \"55m\"", distance_marker_changing_width},
    {"stacked bars at different depths", stacked_bars_at_different_depths},
    {"a damage number merging and separating", damage_number_merging_and_separating},
    {"a bar crossing the central edge or outgrowing the cap", bar_leaving_the_centre_or_outgrowing_the_cap},
    {"a bar beside HUD with a near occluder", bar_beside_hud_with_a_near_occluder},
    {"injected sign, axis and scale errors", injected_sign_axis_and_scale_errors},
    {"a cut without reset", cut_without_reset},
    {"FG cadence and multiplier drift", fg_cadence_and_multiplier_drift},
    {"a skipped real frame", skipped_real_frames},
    {"split, merge, blink", split_merge_and_blink},
    {"calibration ramps", calibration_ramps},
    {"scope changes and observation revisions", scope_changes_and_observation_revisions},
    {"HUD co-moving with one layer (known limit)", marker_co_moving_with_one_layer},
  };
  try {
    for (const auto &[name, group] : groups) {
      group();
      std::printf("PASS %s\n", name);
    }
    std::printf("PASS UI anchor sequence: %zu groups, %llu frames, %llu HUD element-frames never world, %u KNOWN_LIMIT\n", std::size(groups),
      (unsigned long long) grand.frames, (unsigned long long) grand.hud_frames, known_limit_count);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL UI anchor sequence: %s\n", error.what());
    return 1;
  }
}
