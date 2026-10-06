// SPDX-License-Identifier: GPL-3.0-only
// Actual preserve2 D3D12 challenger promotion through native Game 3D with no
// installed FX. No injected selection, depth, readiness or camera.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"

namespace {
  struct probe_round_fixture : raw_runtime_fixture {
    using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
    using state_t = BOOL (*)(api::effect_runtime *, std::uint64_t *, BOOL *);
    select_t select_manual = nullptr;
    state_t manual_state = nullptr;
    native_game3d_observer game3d{*this};
    // What the native render of the latest Present consumed.
    sunshine_game3d::test::last_render frame;
    std::uint64_t rendered_sequence = 0;
    std::vector<std::unique_ptr<target_t>> flats;
    std::unique_ptr<target_t> missing;
    std::array<std::unique_ptr<target_t>,2> lower_peers;
    bool rotating_lower=false;
    unsigned lower_present=0;
    target_t *lower_rendered=nullptr;
    std::ofstream trace;

    bool current(const target_t &target) const {
      return frame.depth.ready &&
        frame.depth.source_resource.handle == reinterpret_cast<std::uint64_t>(target.texture.p) &&
        frame.depth.width == target.width && frame.depth.height == target.height;
    }
    // The current source rendered at full Automatic strength.
    bool full() const {
      return frame.rendered && frame.parameters.depth_ready && frame.parameters.camera_ready &&
        frame.parameters.strength_blend == 1.f;
    }
    void render_depth() {
      if (rotating_lower) {
        const unsigned slot=lower_present++%3;
        lower_rendered=slot ? lower_peers[slot-1].get() : scene.get();
        draw(*lower_rendered);
        if (decoy) draw(*decoy); // New full-resolution challenger renders every present.
        return;
      }
      if (missing) {
        // Real useful raster depth with no legal preserve2 discard capture.
        // The transition invalidates the clear-established direct-switch proof;
        // the next target's null unbind cannot preserve this resource instead.
        draw(*missing, false);
        transition(commands.p, missing->texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      }
      draw(*scene);
      if (missing)
        transition(commands.p, missing->texture.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_DEPTH_WRITE);
      if (decoy) draw(*decoy);
      for (const auto &flat : flats) draw(*flat);
    }
    void tick(const char *phase) {
      step(); game3d.no_effects();
      frame = game3d.last_render();
      require(frame.sequence > rendered_sequence, "A Present had no native Game 3D render to observe");
      rendered_sequence = frame.sequence;
      trace << phase << ',' << GetTickCount64() << ',' << frame.sequence << ','
        << frame.depth.source_id << ',' << frame.depth.source_resource.handle << ','
        << frame.depth.frame_index << ',' << frame.depth.ready << ',' << frame.parameters.camera_ready << ','
        << frame.parameters.depth_scale << ',' << frame.parameters.strength_blend << '\n';
      require(trace.good(), "Cannot record challenger-promotion trajectory");
      if (missing)
        require(frame.depth.source_resource.handle != reinterpret_cast<std::uint64_t>(missing->texture.p),
                "A candidate without a legal current capture displaced useful scene depth");
    }
    void await_current(const char *phase, const target_t &target, unsigned limit_ms) {
      const auto started = GetTickCount64();
      do { tick(phase); } while (!current(target) && GetTickCount64() - started < limit_ms);
      std::printf("MEASURE %s promotion_ms=%llu limit_ms=%u source=%llu\n", phase,
        static_cast<unsigned long long>(GetTickCount64() - started), limit_ms,
        static_cast<unsigned long long>(frame.depth.source_id));
      require(current(target) && GetTickCount64() - started <= limit_ms,
              "Qualified native challenger did not promote within its bounded visit; inspect trajectory and ReShade log");
    }
    // The production Dump 3D capture of one native render: the exact depth
    // allocation it consumed, read through its own artifact.
    void verify_pattern(const char *phase, const target_t &target) {
      const auto consumed = game3d.capture(phase, false, [&] { tick(phase); });
      require(consumed.depth_ready && consumed.source_resource == reinterpret_cast<std::uint64_t>(target.texture.p) &&
        consumed.width == target.width && consumed.height == target.height,
        "Depth pixel oracle has no current expected original source");
      require(consumed.raw_depth.size() == size_t(target.width) * target.height * sizeof(float), "Unexpected depth-plane size");
      for (unsigned y=0; y<18; ++y) for (unsigned x=0; x<32; ++x) {
        const unsigned px=(2*x+1)*target.width/64, py=(2*y+1)*target.height/36;
        const float raw=consumed.depth(px, py);
        const float u=(px+.5f)/target.width, v=(py+.5f)/target.height;
        const float expected=.01f+.02f*(target.pattern==9 ? 1.f-u : u)+.015f*v;
        require(std::isfinite(raw) && std::abs(raw-expected)<2e-6f,
                "Selected native depth contains a flat decoy, wrong source, or wrong spatial pattern");
      }
    }
    void stable(const char *phase, const target_t &target, unsigned milliseconds) {
      const auto until=GetTickCount64()+milliseconds;
      do { tick(phase); require(current(target), "Useful selected source was lost during stable rendering"); }
      while (GetTickCount64()<until);
    }
    void check_rotating_lower_promotion() {
      // Give the per-member raw captures real recurring work before a
      // challenger exists. They must not keep moving the independent
      // histogram-probe deadline and starving a better continuously live input.
      decoy.reset(); missing.reset(); flats.clear();
      scene=target(width/2,height/2,1,14);
      for (auto &peer:lower_peers) peer=target(width/2,height/2,1,14);
      rotating_lower=true; lower_present=0;
      std::printf("DIAGNOSTIC rotating-lower originals A=%llu B=%llu C=%llu\n",
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(scene->texture.p)),
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(lower_peers[0]->texture.p)),
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(lower_peers[1]->texture.p)));
      // The three members draw identical depth, so each member's own
      // controller reaches the same positive scale and screen plane.
      const auto began=GetTickCount64(); unsigned consecutive=0;
      float scale=0.f; std::array<float,2> zero{};
      do {
        tick("rotating-lower-calibration");
        require(!frame.depth.ready || current(*lower_rendered), "Rotating lower-res control admitted another present's member");
        if (current(*lower_rendered) && full()) {
          if (!consecutive) { scale=frame.parameters.depth_scale; zero=frame.parameters.convergence; }
          require(std::isfinite(scale) && scale>0.f, "Rotating lower-res control has no positive automatic scale");
          if (frame.parameters.depth_scale==scale && frame.parameters.convergence==zero) ++consecutive; else consecutive=0;
        } else consecutive=0;
      } while (consecutive<12 && GetTickCount64()-began<15000);
      require(consecutive>=12, "Rotating lower-res sources did not calibrate before the challenger scheduling test");
      std::printf("MEASURE rotating-lower H=%.9g t0=%.9g\n", scale, zero[1]);
      const auto steady_until=GetTickCount64()+1200;
      do {
        tick("rotating-lower-steady");
        require(current(*lower_rendered) && full() && frame.parameters.depth_scale==scale && frame.parameters.convergence==zero,
          "Calibrated rotating sources lost current readiness or their scale before introduction of the native challenger");
      } while (GetTickCount64()<steady_until);
      std::puts("PASS calibrated lower-res ABC remains current and full-strength before challenger introduction");
      decoy=target(width,height,1,9);
      await_current("rotating-raw-does-not-starve-native",*decoy,4000);
      verify_pattern("rotating-raw-does-not-starve-native",*decoy);
      stable("promoted-native-survives-rotating-lower",*decoy,1200);
      verify_pattern("promoted-native-survives-rotating-lower",*decoy);
      std::puts("PASS per-member raw calibration sampling does not starve independent native-depth challenger qualification/promotion");
    }
    void check_native_priority() {
      // All unrelated lower-resolution resources are older than the useful
      // native source. No source has been manually selected or prequalified.
      // The final draw still belongs to a lower-resolution flat, so last-DSV
      // ordering cannot stand in for actual content/resolution discovery.
      for (unsigned i=0;i<12;++i) flats.push_back(target(width/2,height/2,1,0));
      decoy=target(width,height,1,9);
      std::printf("DIAGNOSTIC late-native originals incumbent=%llu native=%llu older_flats=%zu\n",
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(scene->texture.p)),
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(decoy->texture.p)),flats.size());
      await_current("late-native-ahead-of-older-lowres-tour",*decoy,4000);
      verify_pattern("late-native-ahead-of-older-lowres-tour",*decoy);
      stable("late-native-remains-current",*decoy,1200);
      verify_pattern("late-native-remains-current",*decoy);
      std::puts("PASS late-created useful native depth promotes ahead of 12 older active low-resolution flats; actual current source and 576 depth cells verified");

      // Resolution grants a measurement opportunity, never qualification. A
      // subsequent full-resolution flat must not replace the useful fallback.
      decoy.reset(); flats.clear();
      await_current("late-native-destroyed-scaled-recovery",*scene,12000);
      stable("scaled-before-fullres-flat",*scene,1200);
      decoy=target(width,height,1,0);
      stable("fullres-flat-keeps-useful-scaled",*scene,2500);
      verify_pattern("fullres-flat-keeps-useful-scaled",*scene);
      std::puts("PASS full-resolution flat depth cannot displace the qualified lower-resolution current source");
    }
    void run(bool native_priority=false) {
      create_pipeline(); normal=false;
      scene=target(width/2,height/2,1,1);
      render_tracked_depth=[&]{render_depth();};
      trace.open(runtime_directory/"probe-round-trajectory.csv");
      trace << std::setprecision(17) << "phase,wall_ms,render,source,resource,frame,depth_ready,camera_ready,H,blend\n";
      const auto until=GetTickCount64()+45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64()<until) step();
      require(observed.runtime && observed.renders && !observed.inject, "Actual probe-round fixture did not initialize");
      check_unified_addon();
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      select_manual=reinterpret_cast<select_t>(GetProcAddress(module,"SunshineDepthTestSelectManual"));
      manual_state=reinterpret_cast<state_t>(GetProcAddress(module,"SunshineDepthTestManualState"));
      require(select_manual && manual_state,"Probe fixture needs the real manual-selection adapters");
      game3d.start();
      game3d.set_strength(100);
      rendered_sequence=game3d.await_render([&]{ step(); });
      await_current("sole-scaled-startup",*scene,12000);
      const auto calibrated=GetTickCount64()+12000;
      while (!(current(*scene) && full()) && GetTickCount64()<calibrated)
        tick("sole-scaled-calibration");
      require(current(*scene) && full(),
              "Incumbent must be genuinely calibrated before testing challenger scheduling");
      stable("sole-scaled-confirmed",*scene,1200);

      if (native_priority) {
        check_native_priority();
        render_tracked_depth={};
        return;
      }

      // Lifetime order is intentional: useful native first, then six flat but
      // active and otherwise eligible peers. Their unrelated visits must not
      // separate qualification from the remaining fresh paired comparisons.
      decoy=target(width,height,1,9);
      for (unsigned i=0;i<6;++i) flats.push_back(target(width/2,height/2,1,0));
      await_current("native-before-flat-tour",*decoy,3000);
      verify_pattern("native-before-flat-tour",*decoy);
      stable("native-remains-selected",*decoy,1000);
      verify_pattern("native-remains-selected",*decoy);
      std::puts("PASS native source promotes without a full flat-candidate tour; actual source and depth pixels verified");

      // A perpetually uncopyable challenger must still relinquish the bounded
      // probe visit so a later live native source can be considered.
      decoy.reset(); flats.clear();
      await_current("native-destroyed-scaled-recovery",*scene,12000);
      stable("scaled-reconfirmed",*scene,1200);
      missing=target(width,height,1,9);
      decoy=target(width,height,1,9);
      // The uncapturable peer may hold two consecutive discovery visits: the
      // discovery hint's, then the round robin's, which orders candidates by
      // lifetime rather than by the hint. Each visit is a queued and then an
      // allocated capture phase of at most 2 s (sunshine_depth_probe::
      // phase_expired) before the useful peer's own qualifying visit.
      constexpr unsigned probe_phase_ms=2000;
      await_current("uncaptured-candidate-yields",*decoy,2*(2*probe_phase_ms)+probe_phase_ms);
      verify_pattern("uncaptured-candidate-yields",*decoy);
      std::puts("PASS uncaptured active candidate cannot starve a later useful source or become selected");
      missing.reset();

      // An explicitly pinned and continuously rendered source is an ownership
      // boundary, even with a already-confirmed better-resolution alternative.
      const auto pinned=reinterpret_cast<std::uint64_t>(scene->texture.p);
      require(select_manual(observed.runtime,pinned),"Cannot invoke real manual depth-selection action");
      await_current("manual-pin-becomes-current",*scene,1000);
      const auto pin_until=GetTickCount64()+3000;
      do {
        tick("manual-pin-retained");
        std::uint64_t actual=0; BOOL recovering=TRUE;
        require(manual_state(observed.runtime,&actual,&recovering) && actual==pinned && !recovering && current(*scene),
                "Challenger scheduling replaced an active manual depth pin");
      } while (GetTickCount64()<pin_until);
      verify_pattern("manual-pin-retained",*scene);
      require(select_manual(observed.runtime,0),"Cannot return to automatic selection");
      await_current("manual-release-native-recovery",*decoy,12000);
      verify_pattern("manual-release-native-recovery",*decoy);
      std::puts("PASS active manual pin remains authoritative and explicit release restores automatic native selection");
      check_rotating_lower_promotion();
      render_tracked_depth={};
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && (argc!=6 || std::strcmp(argv[5],"--native-priority")!=0)){
    std::fputs("usage: depth_probe_round_runtime <official.dll> <frozenShaders> <test.addon64> <fresh-output> [--native-priority]\n",stderr);return 2;
  }
  std::thread([]{Sleep(150000);std::fputs("FAIL probe-round watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    select_native_boot(true);
    width=3840; height=2160;
    require(!fs::exists(fs::absolute(argv[4])),"Fresh probe-round output required");
    probe_round_fixture f; f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));
    f.run(argc==6); return 0;
  } catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
