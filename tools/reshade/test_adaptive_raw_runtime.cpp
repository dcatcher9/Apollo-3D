// SPDX-License-Identifier: GPL-3.0-only
// Actual DSV -> selector/sampler -> raw-depth scale controller -> native
// Game 3D HDR rendering with no installed FX. No depth, gain or readiness
// injection; the scale and screen-plane oracle is the owning contract's
// formula applied to the independently known draw.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"
#include <deque>

namespace {
  struct scheduling_stall : std::runtime_error { using std::runtime_error::runtime_error; };
  struct adaptive_fixture : raw_runtime_fixture {
    std::ofstream trajectory;
    std::uint64_t last_frame_completed_ms = 0;
    bool require_continuous_frames = false;
    using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
    select_t select_manual = nullptr;
    native_game3d_observer game3d{*this};
    // What the native render of the latest Present consumed.
    sunshine_game3d::test::last_render latest;
    std::uint64_t rendered_sequence = 0;
    // Recent renders, to find the one a Dump 3D capture consumed.
    std::deque<sunshine_game3d::test::last_render> recent;
    unsigned pattern(bool changed) const { return (normal ? 16u : 14u) + unsigned(changed); }
    float gain() const { return latest.parameters.depth_scale; }
    float zero() const { return latest.parameters.convergence[1]; }
    float blend() const { return latest.parameters.strength_blend; }
    bool ready() const { return latest.rendered && latest.parameters.depth_ready && latest.parameters.camera_ready; }

    // docs/reshade-sbs.md, raw-depth automation: oriented raw t (t=d reversed,
    // t=1-d normal) of the coherent bands and fixed center patch; gain L/Q
    // from the full-image maximum Q, zero at the contrast midpoint
    // b+0.5*sum(d*d)/sum(d) with b the minimum and d=t-b.
    static double oriented(unsigned x, unsigned y, unsigned w, unsigned h, bool changed) {
      const unsigned cx = std::min(31u, unsigned((float(x) + .5f) / float(w) * 32.f));
      const unsigned cy = std::min(17u, unsigned((float(y) + .5f) / float(h) * 18.f));
      const bool center = cx >= 14 && cx < 18 && cy >= 7 && cy < 11;
      return center ? (changed ? .25 : .125) : double(cx / 8 + 1) / 16.;
    }
    std::array<double, 2> oracle(bool changed) const {
      double minimum = 1, maximum = 0, sum = 0, square = 0;
      for (unsigned y = 0; y < scene->height; ++y) for (unsigned x = 0; x < scene->width; ++x) {
        const double t = oriented(x, y, scene->width, scene->height, changed);
        minimum = std::min(minimum, t); maximum = std::max(maximum, t);
      }
      for (unsigned y = 0; y < scene->height; ++y) for (unsigned x = 0; x < scene->width; ++x) {
        const double d = oriented(x, y, scene->width, scene->height, changed) - minimum;
        sum += d; square += d * d;
      }
      return {limit() / maximum, minimum + .5 * square / sum};
    }
    // The full-strength normalization L of the output shape.
    double limit() const {
      const double c = (double(height) / 2160. * 100.) / double(width);
      return std::min({std::min(1.5, .04 / c), std::min(2.5, .04 / c), .01 / c}) / .05;
    }
    static bool close(double actual, double expected) {
      return std::isfinite(actual) && std::abs(actual - expected) <= 1e-5 * std::max(1., std::abs(expected));
    }
    void record(const char *phase) {
      trajectory << phase << ',' << GetTickCount64() << ',' << latest.sequence << ',' << latest.depth.source_resource.handle << ','
        << latest.depth.source_id << ',' << latest.depth.frame_index << ',' << ready() << ',' << gain() << ',' << zero() << ',' << blend() << '\n';
      require(trajectory.good(), "Cannot record actual adaptive trajectory");
    }
    void frame(const char *phase) {
      step(); game3d.no_effects();
      latest = game3d.last_render();
      require(latest.sequence > rendered_sequence, "A Present had no native Game 3D render to observe");
      rendered_sequence = latest.sequence;
      recent.push_back(latest);
      if (recent.size() > 64) recent.pop_front();
      record(phase);
      const auto now = GetTickCount64(), previous = last_frame_completed_ms;
      last_frame_completed_ms = now;
      if (require_continuous_frames && previous && now - previous > 250) {
        trajectory.flush();
        throw scheduling_stall(std::string(phase) + ": completed-frame interval " + std::to_string(now - previous) +
          " ms exceeds the 250 ms presentation contract; continuous-adaptation result is inconclusive");
      }
    }
    bool selected_scene() const {
      return latest.depth.ready && latest.depth.source_resource.handle == reinterpret_cast<std::uint64_t>(scene->texture.p);
    }
    template<class Check> void pump(unsigned ms, const char *phase, Check check) {
      const auto end = GetTickCount64() + ms;
      do { frame(phase); check(); } while (GetTickCount64() < end);
    }
    // One production Dump 3D capture: the exact depth and constants a native
    // render consumed, and optionally its packed SBS.
    native_render inspect(const char *phase, bool sbs = false) {
      return game3d.capture(phase, sbs, [&] { frame(phase); });
    }
    void verify_depth(const char *phase, bool changed) {
      const auto consumed = inspect(phase);
      require(consumed.depth_ready && consumed.source_resource == reinterpret_cast<std::uint64_t>(scene->texture.p) &&
        consumed.width == scene->width && consumed.height == scene->height,
        "The native render did not consume the current adaptive scene depth");
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const auto px = (2 * x + 1) * scene->width / 64, py = (2 * y + 1) * scene->height / 36;
        const float raw = consumed.depth(px, py);
        const bool center = x >= 14 && x < 18 && y >= 7 && y < 11;
        const float expected = center ? (changed ? .25f : .125f) : (float(x / 8 + 1) / 16.f);
        require(std::isfinite(raw) && (normal ? 1.f - raw : raw) == expected,
          "Consumed native depth differs from the independent coherent-band/center draw oracle");
      }
      // The same render, as each Present observed it.
      const auto same = std::find_if(recent.rbegin(), recent.rend(), [&](const auto &render) {
        return render.depth.ready && render.depth.frame_index == consumed.frame_index &&
          render.depth.source_resource.handle == consumed.source_resource;
      });
      require(same != recent.rend(), "The captured native render was not among the observed Presents");
      require(consumed.camera_ready && consumed.coordinate_basis == 1 && same->parameters.camera_ready &&
        consumed.depth_scale == same->parameters.depth_scale && consumed.convergence == same->parameters.convergence &&
        consumed.strength_blend == same->parameters.strength_blend,
        "The native render did not consume the adaptive gain and screen plane of its own Present");
    }
    void settle_ready(const char *phase, unsigned timeout = 12000) {
      const auto end = GetTickCount64() + timeout;
      do { frame(phase); } while ((!ready() || !selected_scene() || blend() != 1.f) && GetTickCount64() < end);
      require(ready() && selected_scene() && blend() == 1.f, "Adaptive source did not reach ready full-strength stereo");
    }
    void qualify_source(const char *phase, bool changed, std::uint64_t retired_lifetime = 0) {
      // No draw of this new source/action epoch has occurred before entry.
      // Its samples may arrive before the renderer first selects it, so the
      // fresh-window lower bound belongs to this earliest possible draw, not
      // to first_selected. Keep selection latency as a separate upper bound.
      const auto first_possible_draw = GetTickCount64(), end = first_possible_draw + 10000;
      std::uint64_t first_selected = 0, first_ready = 0;
      do {
        frame(phase);
        if (GetTickCount64() - first_possible_draw < 650)
          require(!ready() && blend() == 0.f, "Source initialization reused old scale before one fresh reference window");
        if (selected_scene()) {
          require(!retired_lifetime || latest.depth.source_id != retired_lifetime,
            "Destroyed source replacement reused the retired lifetime identity");
          if (!first_selected) first_selected = GetTickCount64();
        }
        if (!first_selected) continue;
        const auto elapsed = GetTickCount64() - first_selected;
        if (ready()) { first_ready = GetTickCount64(); break; }
        require(elapsed <= 1400, "Source recovery exceeded one 750 ms reference window plus bounded capture latency");
      } while (GetTickCount64() < end);
      require(first_selected && first_ready && first_ready - first_selected <= 1400,
        "Fresh selected source failed the bounded single-window initialization");
      const auto expected = oracle(changed);
      require(close(gain(), expected[0]) && close(zero(), expected[1]),
        "New source inherited an old reference instead of initializing from its own range and contrast midpoint");
      std::printf("PASS source initialization %s: draw-entry-to-ready=%llu ms selected-to-ready=%llu ms H=%.9g t0=%.9g; one fresh window\n",
        phase, static_cast<unsigned long long>(first_ready - first_possible_draw),
        static_cast<unsigned long long>(first_ready - first_selected), gain(), zero());
      settle_ready(phase);
    }
    void run_adaptive(bool reversed) {
      normal = !reversed;
      trajectory.open(runtime_directory / "adaptive-trajectory.csv");
      trajectory << std::setprecision(17) << "phase,wall_ms,render,source,lifetime,frame,ready,H,t0,blend\n";
      create_pipeline();
      scene = target(width, height, 1, pattern(false), false, true);
      scene->clear_depth = normal ? 1.f : 0.f;
      decoy = target(width / 2, height / 2, 12, 0);
      render_tracked_depth = [&] { if (decoy) draw(*decoy); if (scene) draw(*scene); };
      const auto load_end = GetTickCount64() + 45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64() < load_end) step();
      require(observed.runtime && observed.renders && !observed.inject, "Actual adaptive fixture did not initialize without injected depth");
      check_unified_addon();
      const auto module = GetModuleHandleW(L"SunshineSBSTest.addon64");
      select_manual = reinterpret_cast<select_t>(GetProcAddress(module, "SunshineDepthTestSelectManual"));
      require(select_manual, "Adaptive fixture requires the test-only real UI-selection adapter");
      game3d.start();
      game3d.set_strength(100);
      rendered_sequence = game3d.await_render([&] { step(); });
      settle_ready("initial");
      const auto initial = oracle(false), changed_scene = oracle(true);
      std::printf("MEASURE adaptive oracle constant H=%.9g t0=%.9g changed t0=%.9g\n", initial[0], initial[1], changed_scene[1]);
      require(close(gain(), initial[0]) && close(zero(), initial[1]),
        "Constant scene did not initialize gain L/Q and its contrast-midpoint screen plane");
      verify_depth("initial-consumed", false);
      settle_ready("after-initial-readback");
      const auto binding = latest.depth.resource.handle;
      const float constant_gain = gain(), constant_zero = zero();
      const auto constant = [&] { require(ready() && blend() == 1.f && gain() == constant_gain && zero() == constant_zero &&
        latest.depth.resource.handle == binding, "Constant-source pin/unpin changed scale, zero, binding or ready strength"); };
      require(select_manual(observed.runtime, reinterpret_cast<std::uint64_t>(scene->texture.p)), "Cannot pin actual adaptive source");
      pump(1200, "constant-pin", constant);
      require(select_manual(observed.runtime, 0), "Cannot release actual adaptive source");
      pump(1200, "constant-unpin", constant);
      std::printf("PASS constant scene and same-source pin/unpin retain H=%.9g, t0=%.9g, binding, readiness and full blend\n",
        constant_gain, constant_zero);

      // Only the center patch changes: the nearest reference Q, hence the
      // gain target, is unchanged while the zero follows the new contrast
      // midpoint with exponential smoothing, never moving away from it.
      scene->pattern = pattern(true);
      const auto started = GetTickCount64();
      require_continuous_frames = true;
      double distance = std::abs(double(zero()) - changed_scene[1]);
      pump(7000, "zero-plane-changed-scene", [&] {
        require(ready() && blend() == 1.f && selected_scene() && gain() == constant_gain,
          "Changed center patch moved the gain or lost readiness/source");
        const double now_distance = std::abs(double(zero()) - changed_scene[1]);
        require(now_distance <= distance + 1e-7, "Screen plane moved away from the changed scene's contrast midpoint");
        distance = now_distance;
        if (GetTickCount64() - started >= 5000)
          require(close(zero(), changed_scene[1]), "Independent screen plane did not settle at the changed scene's contrast midpoint");
      });
      require_continuous_frames = false;
      std::printf("PASS screen plane %.9g -> %.9g with gain held at %.9g; full blend retained\n", constant_zero, zero(), gain());
      verify_depth("changed-consumed", true);
      settle_ready("after-adaptive-readback");

      const float before_gap = gain(), zero_before_gap = zero();
      const auto gap_started = GetTickCount64();
      render_tracked_depth = {};
      const auto missing_publication = [&] {
        // A mono render carries no scene constants; the controller holds the
        // established gain (its automatic scale) and zero. A pending valid
        // readback may update the zero target but cannot move the plane: the
        // first ready render after the gap shows the held zero.
        const auto status = game3d.automatic();
        require(!latest.depth.ready && !ready() && blend() == 0.f && !status.ready() &&
          status.scale_state == unsigned(sunshine_game3d::automatic_scale_state::held) && status.scale == before_gap,
          "Missing current depth rendered stereo or changed the held gain");
      };
      pump(100, "depth-gap-drain", missing_publication);
      // Change visible source color during the depth gap, so stale output cannot pass.
      for (size_t i = 0; i < size_t(width) * height; ++i) {
        if (color == 2) { const std::uint16_t red = 0x3a00; std::memcpy(source_bytes.data() + i * 8, &red, 2); }
        else { std::uint32_t pixel = 0; std::memcpy(&pixel, source_bytes.data() + i * 4, 4); pixel = (pixel & ~1023u) | 450u; std::memcpy(source_bytes.data() + i * 4, &pixel, 4); }
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      const auto gap = inspect("depth-gap-new-color", true);
      require(!gap.depth_ready && !gap.camera_ready, "A depth gap rendered with stale depth");
      check_current_mono(gap.sbs);
      missing_publication();
      // No depth draws means no new selected captures. Continue presenting past
      // the 1500 ms evidence expiry: a stale target/duplicate must not accumulate
      // hidden gain that would be revealed on the next ready publication.
      pump(1600, "depth-gap-no-new-samples", missing_publication);
      render_tracked_depth = [&] { if (scene) draw(*scene); };
      const auto first_ready_until = GetTickCount64() + 12000;
      do { frame("depth-return-first-ready"); } while (!ready() && GetTickCount64() < first_ready_until);
      require(ready() && selected_scene(), "Depth return failed to render a fresh current scene");
      require(gain() == before_gap, "Missing captures or their return changed the held gain");
      require(zero() == zero_before_gap, "First ready depth return spent missing-time zero-plane credit");
      std::printf("PASS gap hold: elapsed=%llu ms gain=%.9g -> %.9g; zero held\n",
        static_cast<unsigned long long>(GetTickCount64() - gap_started), before_gap, gain());
      settle_ready("depth-return");
      const float before_pause = gain();
      Sleep(650);
      frame("presentation-gap-first");
      require(gain() == before_pause && blend() == 0.f, "Presentation gap accumulated gain/reentry credit");
      settle_ready("presentation-return");
      std::puts("PASS current-color mono while unavailable, held gain, and no missing/presentation-time zero catch-up credit");

      // Save the last actual A render immediately before leaving it. The
      // still-live exact source keeps its own numerical history off-turn.
      require(ready() && selected_scene() && blend() == 1.f, "Cannot save an unready A reference");
      const auto original_lifetime = latest.depth.source_id;
      const float original_gain = gain(), original_zero = zero();
      auto original = std::move(scene);
      scene = target(width, height, 1, pattern(true), false, true); scene->clear_depth = normal ? 1.f : 0.f;
      qualify_source("replacement-B", true);
      auto replacement = std::move(scene);
      scene = std::move(original); scene->pattern = pattern(false);
      const auto return_end = GetTickCount64() + 10000;
      std::uint64_t return_selected = 0;
      do {
        frame("return-A-retained-reference");
        if (selected_scene() && !return_selected) return_selected = GetTickCount64();
        if (ready() && selected_scene()) break;
        if (return_selected) require(GetTickCount64() - return_selected <= 1400,
          "Retained A did not resume within bounded fresh-capture latency");
      } while (GetTickCount64() < return_end);
      require(return_selected && ready() && selected_scene() && GetTickCount64() - return_selected <= 1400 &&
        latest.depth.source_id == original_lifetime, "Retained A did not resume its exact live source");
      require(gain() == original_gain, "Retained A reset its gain while off-turn");
      require(zero() == original_zero, "Retained A reset its zero or spent off-turn motion credit");
      std::printf("PASS retained A: lifetime=%llu H=%.9g -> %.9g t0=%.9g; prior basis resumed, no motion catch-up\n",
        static_cast<unsigned long long>(original_lifetime), original_gain, gain(), zero());
      settle_ready("return-A-full-strength");
      verify_depth("return-A-consumed", false);
      // Actual destruction must retire that history even if the native address
      // is reused by D3D12. The new lifetime still needs one fresh window.
      scene.reset();
      scene = target(width, height, 1, pattern(false), false, true); scene->clear_depth = normal ? 1.f : 0.f;
      qualify_source("destroyed-A-replacement", false, original_lifetime);
      require(latest.depth.source_id != original_lifetime, "Destroyed A replacement reused the old lifetime identity");
      verify_depth("destroyed-A-replacement-consumed", false);
      settle_ready("before-recalibrate");
      scene->pattern = pattern(true);
      pump(1000, "before-recalibrate-changed", [&] { require(ready() && blend() == 1.f, "Pre-action adaptation lost readiness"); });
      require(game3d.recalibrate() && !game3d.recalibrate(), "Recalibrate did not admit exactly one pending real UI action");
      qualify_source("explicit-recalibrate", true);
      verify_depth("explicit-recalibrate-consumed", true);
      settle_ready("final");
      game3d.set_strength(0); pump(80, "zero-strength", [] {});
      const auto mono = check_current_mono(inspect("zero-strength", true).sbs);
      game3d.set_strength(100); settle_ready("stereo-restored");
      const auto stereo = inspect("stereo-restored", true);
      require(stereo.camera_ready && stereo.strength_blend == 1.f, "Restored stereo render was not at full strength");
      for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width * 2; ++x) for (unsigned c = 0; c < 4; ++c)
        require(std::isfinite(channel(stereo.sbs, x, y, c)), "Adaptive HDR SBS contains nonfinite RGBA");
      require(image_difference(mono, stereo.sbs) > .002f, "Adaptive ready state did not produce actual stereo pixels");
      sunshine_parity::write_bytes(runtime_directory / "adaptive-final.sbs", stereo.sbs.data(), stereo.sbs.size());
      sunshine_parity::write_bytes(runtime_directory / "adaptive-current-source.bin", source_bytes.data(), source_bytes.size());
      std::puts("PASS actual raw-depth HDR through native Game 3D: L/Q gain and contrast-midpoint zero, current mono, retained exact source, fresh lifetime/recalibration and finite stereo; no FX");
      render_tracked_depth = {};
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 9) { std::fputs("usage: adaptive_raw_runtime <official.dll> <frozenShaders> <test.addon64> <fresh-output> <scrgb|pq> <normal|reversed> width height\n", stderr); return 2; }
  std::thread([] { Sleep(180000); std::fputs("FAIL adaptive runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    width = unsigned(std::stoul(argv[7])); height = unsigned(std::stoul(argv[8]));
    require(width >= 640 && width <= 3840 && height >= 360 && height <= 2160 && width % 2 == 0 && height % 2 == 0, "Invalid adaptive fixture dimensions");
    require(std::string(argv[5]) == "scrgb" || std::string(argv[5]) == "pq", "Invalid adaptive fixture color");
    require(std::string(argv[6]) == "normal" || std::string(argv[6]) == "reversed", "Invalid adaptive depth convention");
    // The FX reload case covered the retired effect's own scale reset; native
    // capture and scale deliberately survive an FX reload
    // (reshade_game3d_native_depth_runtime_test).
    require(!sunshine_camera_fixture::flag("SUNSHINE_DEPTH_RELOAD_TEST"),
      "SUNSHINE_DEPTH_RELOAD_TEST is retired: native capture and scale survive an FX reload");
    select_native_boot();
    require(!fs::exists(fs::absolute(argv[4])), "Adaptive fixture requires a fresh isolated output");
    adaptive_fixture fixture; fixture.runtime_directory = fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fixture.runtime_directory, std::string(argv[5]) == "pq" ? 3 : 2, 0, fs::absolute(argv[3]));
    fixture.run_adaptive(std::string(argv[6]) == "reversed");
    return 0;
  } catch (const scheduling_stall &error) { std::fprintf(stderr, "INCONCLUSIVE %s\n", error.what()); return 3; }
  catch (const std::exception &error) { std::fprintf(stderr, "FAIL %s\n", error.what()); return 1; }
}
