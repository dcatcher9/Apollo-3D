// SPDX-License-Identifier: GPL-3.0-only
// Preserve-mode2 actual D3D12 A->B switching through native Game 3D with no
// installed FX. No depth/readiness/camera injection.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"

namespace {
  struct bind_switch_fixture : raw_runtime_fixture {
    enum class sequence { null_unbind, direct_switch, same_bind, no_work, transitioned, unknown_alias, missing };
    sequence order = sequence::null_unbind;
    native_game3d_observer game3d{*this};
    std::ofstream trace;
    unsigned presents = 0;

    void render_depth() {
      if (order == sequence::missing) return;
      const auto a = scene->heap->GetCPUDescriptorHandleForHeapStart();
      if (order == sequence::no_work) {
        commands->OMSetRenderTargets(0, nullptr, FALSE, &a);
        commands->ClearDepthStencilView(a, D3D12_CLEAR_FLAG_DEPTH, 0.f, 0, 0, nullptr);
      } else {
        draw(*scene, order == sequence::null_unbind);
        if (order == sequence::same_bind) commands->OMSetRenderTargets(0, nullptr, FALSE, &a);
        if (order == sequence::transitioned)
          transition(commands.p, scene->texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        if (order == sequence::unknown_alias) {
          // A null/null alias barrier is legal without naming committed A/B as
          // aliased resources. The callback cannot identify what was invalidated;
          // the add-on must discard its proof rather than guess safe copy state.
          D3D12_RESOURCE_BARRIER barrier {};
          barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
          commands->ResourceBarrier(1, &barrier);
        }
      }
      // B's final null bind must never stand in for preserving A at A->B.
      draw(*decoy);
      if (order == sequence::transitioned)
        transition(commands.p, scene->texture.p, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    bool current_a(const native_render &frame) const {
      return frame.depth_ready && frame.source_resource == reinterpret_cast<std::uint64_t>(scene->texture.p) &&
        frame.width == scene->width && frame.height == scene->height && !frame.x && !frame.y &&
        frame.active_width == scene->width && frame.active_height == scene->height;
    }
    automatic_status tick(const char *phase) {
      step(); game3d.no_effects();
      const auto status = game3d.automatic();
      trace << phase << ',' << GetTickCount64() << ',' << ++presents << ",present," << status.ready() << ',' << status.scale_state << ','
        << status.scale << ",,,,,,,,\n";
      require(trace.good(), "Cannot record bind-switch trajectory");
      return status;
    }
    // One captured native render, presenting through tick() until it completes.
    native_render inspect(const char *phase, bool sbs = false, const std::function<void(const automatic_status &)> &each = {}) {
      const auto frame = game3d.capture(phase, sbs, [&] { const auto status = tick(phase); if (each) each(status); });
      trace << phase << ',' << GetTickCount64() << ',' << presents << ",render,,,," << frame.source_resource << ',' << frame.source_id << ','
        << frame.frame_index << ',' << frame.depth_ready << ',' << frame.camera_ready << ',' << frame.depth_scale << ','
        << frame.convergence[1] << ',' << frame.strength_blend << '\n';
      require(trace.good(), "Cannot record bind-switch trajectory");
      return frame;
    }
    void verify_pattern(const native_render &frame, unsigned pattern) {
      require(current_a(frame), "Current rendered A has no ready preserved depth copy");
      // The renderer's SRV covers the preserved D32S8 allocation; its artifact
      // is the unfiltered depth plane, never packed stencil or a B resource.
      require(frame.allocation_format == unsigned(api::format::r32_g8_typeless) ||
        frame.allocation_format == unsigned(api::format::d32_float_s8_uint), "Bind-switch oracle lost actual D32S8 geometry/format");
      require(frame.raw_depth.size() == size_t(scene->width) * scene->height * sizeof(float), "Unexpected D32S8 depth-plane size");
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const unsigned px = (2*x+1)*scene->width/64, py = (2*y+1)*scene->height/36;
        const float raw = frame.depth(px, py);
        const float u = (px+.5f)/scene->width, v = (py+.5f)/scene->height;
        float expected = .01f + .02f*(pattern == 9 ? 1.f-u : u) + .015f*v;
        if (pattern == 14) {
          const unsigned cx = std::min(31u, unsigned(u*32)), cy = std::min(17u, unsigned(v*18));
          expected = cx >= 14 && cx < 18 && cy >= 7 && cy < 11 ? .125f : float(cx/8+1)/16.f;
        }
        require(std::isfinite(raw) && std::abs(raw-expected) < 2e-6f,
          "Preserved A contains a stale pattern, B depth, or invalid native depth");
      }
    }
    native_render settle(const char *phase, bool sbs = false, unsigned limit_ms = 12000) {
      const auto until = GetTickCount64()+limit_ms;
      do {
        if (!tick(phase).ready()) continue;
        // Readiness alone is not the full-strength re-entry; inspect the render.
        auto frame = inspect(phase, sbs);
        if (current_a(frame) && frame.camera_ready && frame.strength_blend == 1.f) return frame;
      } while (GetTickCount64()<until);
      throw std::runtime_error("Bind-switch source did not reach full Automatic readiness");
    }
    void dynamic_copies(const char *phase, sequence next) {
      order = next;
      for (unsigned i=0; i<8; ++i) {
        scene->pattern = i%2 ? 9 : 1;
        const auto frame = inspect(phase, false, [](const automatic_status &status) {
          require(status.ready(), "Continuous captured A interrupted Automatic calibration/readiness");
        });
        require(current_a(frame), "Continuous A rendering lost capture at a DSV binding boundary");
        require(frame.camera_ready && frame.strength_blend == 1.f, "Continuous captured A interrupted Automatic calibration/readiness");
        verify_pattern(frame, scene->pattern);
      }
      std::printf("PASS %s: eight changing actual D32S8 depth patterns captured from current A\n", phase);
    }
    void unavailable(const char *phase, sequence next) {
      order = next;
      const auto refuse = [](const automatic_status &status) {
        require(!status.ready(), "No-work/missing/transitioned/aliased A falsely authorized current depth or stereo");
      };
      for (unsigned i=0; i<3; ++i) refuse(tick(phase));
      const auto frame = inspect(phase, true, refuse);
      require(!current_a(frame) && !frame.camera_ready && frame.strength_blend == 0.f,
        "No-work/missing/transitioned/aliased A falsely authorized current depth or stereo");
      check_current_mono(frame.sbs);
      std::printf("PASS %s: no current A; native output is current-color mono\n", phase);
      order = sequence::direct_switch; scene->pattern=14;
      verify_pattern(settle("valid-direct-recovery"), 14);
    }
    void run() {
      create_pipeline(); normal=false;
      scene=target(2228,1256,4,14,true,false);
      decoy=target(640,360,1,0,true,false);
      render_tracked_depth=[&]{render_depth();};
      trace.open(runtime_directory/"bind-switch-trajectory.csv");
      trace << std::setprecision(17) << "phase,wall_ms,present,kind,automatic_ready,scale_state,K,source,source_id,frame,capture_ready,camera_ready,H,t0,blend\n";
      const auto until=GetTickCount64()+45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64()<until) step();
      require(observed.runtime && observed.renders && !observed.inject, "Actual bind-switch fixture did not initialize");
      check_unified_addon();
      if (sunshine_camera_fixture::flag("SUNSHINE_DEPTH_GENERIC_ONLY_TEST")) {
        // The shared writer puts these in the actual ReShade.ini BEFORE the
        // runtime/add-on DLL loads. Validate what the runtime parsed; do not
        // turn providers off after initialization through a test-only setter.
        for (const char *key : {"StreamlineDepthSource", "NGXDepthSource", "StreamlineCameraProbe", "UpscalerCallTrace"}) {
          unsigned enabled = 1;
          require(reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", key, enabled) && enabled == 0,
            "Generic-only fixture did not disable every provider/probe/trace in the actual runtime config");
        }
        std::puts("PASS Generic-only configuration: Streamline, NGX, camera probe and call trace disabled before initialization");
      }
      game3d.start();
      game3d.set_strength(100);
      const auto baseline=settle("null-unbind-startup");verify_pattern(baseline,14);
      const auto baseline_scale=game3d.automatic();
      std::printf("MEASURE null-unbind baseline H=%.9g t0=%.9g K=%.9g basis=%u\n",baseline.depth_scale,baseline.convergence[1],baseline_scale.scale,baseline_scale.basis);
      require(baseline.coordinate_basis==1 && baseline_scale.basis==unsigned(sunshine_game3d::automatic_scale_basis::relative_depth) &&
        baseline_scale.active_scale() && std::isfinite(baseline.depth_scale) && baseline.depth_scale>0.f,
        "Null-unbind baseline did not initialize a relative-depth scale from actual depth");
      dynamic_copies("null-unbind-control",sequence::null_unbind);
      dynamic_copies("direct-A-to-B",sequence::direct_switch);
      dynamic_copies("same-A-then-B",sequence::same_bind);
      // A->A is not a discard; A->B is. Clearing/rebinding with no draw is not
      // evidence of scene depth. A transitioned out of DEPTH_WRITE cannot be
      // captured with an invented DEPTH_WRITE->COPY_SOURCE barrier.
      unavailable("clear-only-A-to-B",sequence::no_work);
      unavailable("A-transitioned-before-B",sequence::transitioned);
      unavailable("unknown-alias-before-B",sequence::unknown_alias);
      unavailable("actual-missing-depth",sequence::missing);
      order=sequence::direct_switch;scene->pattern=14;
      require(game3d.recalibrate(),"Cannot request real Automatic recalibration");
      const auto started=GetTickCount64(); bool saw_unready=false, ready_during=false;
      const auto refreshing=[&](const automatic_status &status) {
        saw_unready |= !status.ready(); ready_during |= status.ready();
        if (GetTickCount64()-started<650) require(!status.ready(),"Direct-switch recalibration reused the old reference");
      };
      const auto fresh=inspect("direct-switch-fresh-reference",false,refreshing);
      require(current_a(fresh),"Direct-switch reference lost current native A");
      require(ready_during || !fresh.camera_ready,"Direct-switch recalibration rendered the old reference");
      automatic_status status=game3d.automatic();
      while (!status.ready() && GetTickCount64()-started<1400) { status=tick("direct-switch-fresh-reference"); refreshing(status); }
      std::printf("MEASURE direct-switch recalibration ready_after_ms=%llu\n",static_cast<unsigned long long>(GetTickCount64()-started));
      require(saw_unready && status.ready(),"Direct-switch Automatic reference never initialized from fresh actual depth");
      const auto final_frame=settle("direct-switch-final");verify_pattern(final_frame,14);
      std::printf("MEASURE direct-switch reference H=%.9g t0=%.9g\n",final_frame.depth_scale,final_frame.convergence[1]);
      require(final_frame.depth_scale==baseline.depth_scale && final_frame.convergence==baseline.convergence,
        "Direct-switch fresh reference differs from the null-unbind reference for identical actual depth");
      game3d.set_strength(0);
      const auto mono=check_current_mono(inspect("current-mono",true).sbs);
      // The complete CPU mono/color check can exceed the presentation-gap limit.
      // Resume actual frames through the normal fresh-target/full-blend recovery
      // before asserting stereo; never bypass the add-on's pause protection.
      game3d.set_strength(100);const auto stereo=settle("restored-stereo",true).sbs;
      require(image_difference(mono,stereo)>.002f,"Ready direct-switch camera did not produce actual stereo");
      for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width*2;++x) for(unsigned c=0;c<4;++c)
        require(std::isfinite(channel(stereo,x,y,c)),"Direct-switch stereo contains nonfinite RGBA");
      std::puts("PASS real preserve2 A-to-B through native Game 3D: changing current D32S8 copies, null control, same binding, no-work/state/alias/missing guards, fresh Automatic reference and HDR stereo; no FX");
      render_tracked_depth={};
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5){std::fputs("usage: depth_bind_switch_runtime <official.dll> <Shaders> <test.addon64> <fresh-output>\n",stderr);return 2;}
  std::thread([]{Sleep(180000);std::fputs("FAIL bind-switch watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    select_native_boot(true);
    width=3840;height=2160;
    require(!fs::exists(fs::absolute(argv[4])),"Fresh bind-switch output required");
    bind_switch_fixture f;f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));f.run();return 0;
  } catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
