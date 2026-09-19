// SPDX-License-Identifier: GPL-3.0-only
// Preserve-mode2 actual D3D12 A->B switching. No depth/readiness/camera injection.
#include "test_raw_runtime_fixture.h"
#include "depth_addon.h"

namespace {
  using frame_query_t = BOOL (*)(api::effect_runtime *, sunshine_depth::frame_depth *);
  frame_query_t query_frame = nullptr;
  sunshine_depth::frame_depth captured_frame;
  bool captured_ready = false;
  unsigned captured_render = 0;
  void observe_switch_depth(api::effect_runtime *runtime, api::effect_technique technique,
      api::command_list *, api::resource_view, api::resource_view) {
    char name[256] {}; runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) return;
    captured_frame = {}; captured_ready = query_frame && query_frame(runtime, &captured_frame) && captured_frame.ready;
    captured_render = observed.renders;
  }

  struct bind_switch_fixture : raw_runtime_fixture {
    enum class sequence { null_unbind, direct_switch, same_bind, no_work, transitioned, unknown_alias, missing };
    sequence order = sequence::null_unbind;
    using action_t = BOOL (*)(api::effect_runtime *);
    action_t recalibrate = nullptr;
    std::ofstream trace;

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
    bool current_a() const {
      return captured_render == observed.renders && captured_ready &&
        captured_frame.source_resource.handle == reinterpret_cast<std::uint64_t>(scene->texture.p) &&
        captured_frame.width == scene->width && captured_frame.height == scene->height;
    }
    void tick(const char *phase) {
      step();
      trace << phase << ',' << GetTickCount64() << ',' << observed.renders << ',' << captured_frame.source_id << ','
        << captured_frame.frame_index << ',' << captured_ready << ',' << ready() << ','
        << scalar("Sunshine_CameraDepthScale") << ',' << zero()[1] << ',' << scalar("Sunshine_CameraStrengthBlend") << '\n';
      require(trace.good(), "Cannot record bind-switch trajectory");
    }
    void verify_pattern(unsigned pattern) {
      require(current_a(), "Current rendered A has no ready preserved depth copy");
      const auto selected = selected_binding();
      require(selected.handle != 0, "Captured A has no shader-visible depth binding");
      auto *resource = reinterpret_cast<ID3D12Resource *>(observed.runtime->get_device()->get_resource_from_view(selected).handle);
      require(resource, "Captured A has no native depth resource");
      const auto desc = resource->GetDesc();
      require(desc.Width == scene->width && desc.Height == scene->height &&
        (desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || desc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT),
        "Bind-switch oracle lost actual D32S8 geometry/format");
      // Subresource0 is the actual float depth plane; existing read() validates
      // its native footprint and row size instead of assuming packed stencil.
      const auto bytes = read(resource, D3D12_RESOURCE_STATE_COPY_DEST);
      require(bytes.size() == size_t(scene->width) * scene->height * sizeof(float), "Unexpected D32S8 depth-plane size");
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const unsigned px = (2*x+1)*scene->width/64, py = (2*y+1)*scene->height/36;
        float raw = 0; std::memcpy(&raw, bytes.data() + (size_t(py)*scene->width+px)*4, 4);
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
    void settle(const char *phase, unsigned limit_ms = 12000) {
      const auto until = GetTickCount64()+limit_ms;
      do { tick(phase); } while ((!current_a() || !ready() || scalar("Sunshine_CameraStrengthBlend") != 1.f) && GetTickCount64()<until);
      require(current_a() && ready() && scalar("Sunshine_CameraStrengthBlend") == 1.f, "Bind-switch source did not reach full Automatic readiness");
    }
    void dynamic_copies(const char *phase, sequence next) {
      order = next;
      for (unsigned i=0; i<8; ++i) {
        scene->pattern = i%2 ? 9 : 1;
        tick(phase);
        require(current_a(), "Continuous A rendering lost capture at a DSV binding boundary");
        require(ready(), "Continuous captured A interrupted Automatic calibration/readiness");
        verify_pattern(scene->pattern);
      }
      std::printf("PASS %s: eight changing actual D32S8 depth patterns captured from current A\n", phase);
    }
    void unavailable(const char *phase, sequence next) {
      order = next;
      for (unsigned i=0; i<3; ++i) {
        tick(phase);
        require(!current_a() && !ready() && scalar("Sunshine_CameraStrengthBlend") == 0.f,
          "No-work/missing/transitioned A falsely authorized current depth or stereo");
      }
      check_current_mono();
      order = sequence::direct_switch; scene->pattern=14;
      settle("valid-direct-recovery"); verify_pattern(14);
    }
    void run() {
      create_pipeline(); normal=false;
      scene=target(2228,1256,4,14,true,false);
      decoy=target(640,360,1,0,true,false);
      render_tracked_depth=[&]{render_depth();};
      trace.open(runtime_directory/"bind-switch-trajectory.csv");
      trace << std::setprecision(17) << "phase,wall_ms,render,source,frame,capture_ready,automatic_ready,H,t0,blend\n";
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
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      query_frame=reinterpret_cast<frame_query_t>(GetProcAddress(module,"SunshineDepthTestFrame"));
      recalibrate=reinterpret_cast<action_t>(GetProcAddress(module,"SunshineGame3DTestRecalibrate"));
      require(query_frame && recalibrate,"Bind-switch fixture requires current-frame and recalibration test adapters");
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_switch_depth);
      set_int("Depth_Map_View",0);set_float("Depth_Adjustment",100);set_float("Sharpen_Power",0);
      find_texture("DoubleTex",exported,width*2,DXGI_FORMAT_R16G16B16A16_FLOAT);
      settle("null-unbind-startup");verify_pattern(14);
      require(scalar("Sunshine_CameraDepthScale")==8.f && zero()[1]==.125f,"Null-unbind baseline did not initialize from actual center depth");
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
      require(recalibrate(observed.runtime),"Cannot request real Automatic recalibration");
      const auto started=GetTickCount64(); bool saw_unready=false;
      do {
        tick("direct-switch-fresh-reference");
        require(current_a(),"Direct-switch reference lost current native A");
        saw_unready |= !ready();
        if (GetTickCount64()-started<650) require(!ready(),"Direct-switch recalibration reused the old reference");
      } while (!ready() && GetTickCount64()-started<1400);
      require(saw_unready && ready() && scalar("Sunshine_CameraDepthScale")==8.f && zero()[1]==.125f,
        "Direct-switch Automatic reference never initialized from fresh actual depth");
      settle("direct-switch-final");verify_pattern(14);
      set_float("Depth_Adjustment",0);tick("current-mono");const auto mono=check_current_mono();
      // The complete CPU mono/color check can exceed the presentation-gap limit.
      // Resume actual frames through the normal fresh-target/full-blend recovery
      // before asserting stereo; never bypass the add-on's pause protection.
      set_float("Depth_Adjustment",100);settle("restored-stereo");const auto stereo=read(exported.p);
      require(image_difference(mono,stereo)>.002f,"Ready direct-switch camera did not produce actual stereo");
      for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width*2;++x) for(unsigned c=0;c<4;++c)
        require(std::isfinite(channel(stereo,x,y,c)),"Direct-switch stereo contains nonfinite RGBA");
      std::puts("PASS real preserve2 A-to-B: changing current D32S8 copies, null control, same binding, no-work/state/alias/missing guards, fresh Automatic reference and HDR stereo");
      render_tracked_depth={};
      reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_switch_depth);
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5){std::fputs("usage: depth_bind_switch_runtime <official.dll> <Shaders> <test.addon64> <fresh-output>\n",stderr);return 2;}
  std::thread([]{Sleep(180000);std::fputs("FAIL bind-switch watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    require(sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST") &&
      sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST"),
      "Fixture requires explicit preserve2 binding-switch and Automatic test-action flags");
    width=3840;height=2160;
    require(!fs::exists(fs::absolute(argv[4])),"Fresh bind-switch output required");
    bind_switch_fixture f;f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));f.run();return 0;
  } catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
