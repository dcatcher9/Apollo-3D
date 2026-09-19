// SPDX-License-Identifier: GPL-3.0-only
// Mode1 fallback-only capture versus trailing-clear preservation, actual D3D12.
#include "test_raw_runtime_fixture.h"
#include "depth_addon.h"

namespace {
  using frame_query_t = BOOL (*)(api::effect_runtime *, sunshine_depth::frame_depth *);
  frame_query_t query_frame = nullptr;
  sunshine_depth::frame_depth captured;
  unsigned captured_render = 0;
  bool captured_ready = false;
  void observe_mode1(api::effect_runtime *runtime, api::effect_technique technique,
      api::command_list *, api::resource_view, api::resource_view) {
    char name[256] {}; runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) return;
    captured = {};
    captured_ready = query_frame && query_frame(runtime, &captured) && captured.ready;
    captured_render = observed.renders;
  }

  struct mode1_fixture : raw_runtime_fixture {
    std::ofstream trace;
    bool drawing = true;
    bool current() const {
      return captured_ready && captured_render == observed.renders &&
        captured.source_resource.handle == reinterpret_cast<std::uint64_t>(scene->texture.p) &&
        captured.width == scene->width && captured.height == scene->height &&
        captured.x == 0 && captured.y == 0 && captured.active_width == scene->width &&
        captured.active_height == scene->height && captured.aligned_viewport_assumed;
    }
    void tick(const char *phase) {
      step();
      trace << phase << ',' << GetTickCount64() << ',' << observed.renders << ',' << captured.source_id << ','
        << captured.source_resource.handle << ',' << captured.frame_index << ',' << captured_ready << ',' << ready() << ','
        << scalar("Sunshine_CameraDepthScale") << ',' << zero()[1] << ',' << scalar("Sunshine_CameraStrengthBlend") << '\n';
      require(trace.good(), "Cannot record mode1 current-depth trajectory");
    }
    void settle(const char *phase, unsigned timeout = 12000) {
      const auto started = GetTickCount64();
      do { tick(phase); }
      while ((!current() || !ready() || scalar("Sunshine_CameraStrengthBlend") != 1.f) && GetTickCount64() - started < timeout);
      std::printf("MEASURE %s elapsed_ms=%llu current=%d ready=%d H=%.9g blend=%.9g\n", phase,
        static_cast<unsigned long long>(GetTickCount64() - started), current(), ready(),
        scalar("Sunshine_CameraDepthScale"), scalar("Sunshine_CameraStrengthBlend"));
      require(current() && ready() && scalar("Sunshine_CameraStrengthBlend") == 1.f,
        "Mode1 useful current depth never reached actual full-strength Automatic rendering");
    }
    void verify_depth() {
      require(current(), "Mode1 readback has no current expected native source");
      const auto binding = selected_binding();
      auto *resource = reinterpret_cast<ID3D12Resource *>(observed.runtime->get_device()->get_resource_from_view(binding).handle);
      require(resource && resource->GetDesc().Width == scene->width && resource->GetDesc().Height == scene->height,
        "Mode1 backup has unexpected source geometry");
      const auto bytes = read(resource, D3D12_RESOURCE_STATE_COPY_DEST);
      require(bytes.size() == size_t(scene->width) * scene->height * 4, "Unexpected mode1 depth-plane byte count");
      for (unsigned y = 0; y < 18; ++y) for (unsigned x = 0; x < 32; ++x) {
        const unsigned px = (2*x+1)*scene->width/64, py = (2*y+1)*scene->height/36;
        float raw = 0; std::memcpy(&raw, bytes.data() + (size_t(py)*scene->width + px)*4, 4);
        const float u = (px+.5f)/scene->width, v = (py+.5f)/scene->height;
        const bool center = x >= 14 && x < 18 && y >= 7 && y < 11;
        const float expected = scene->pattern == 14 ? (center ? .125f : float(x/8+1)/16.f) :
          .01f + .02f*(scene->pattern == 9 ? 1.f-u : u) + .015f*v;
        require(std::isfinite(raw) && std::abs(raw-expected) < 2e-6f,
          "Mode1 current backup contains a stale, cleared or incorrect spatial depth pattern");
      }
    }
    void run(bool preserved) {
      create_pipeline();
      scene = target(width/2, height/2, 1, 14, false, preserved);
      render_tracked_depth = [&] { if (drawing) draw(*scene); };
      trace.open(runtime_directory/"mode1-trajectory.csv");
      trace << std::setprecision(17) << "phase,wall_ms,render,lifetime,resource,frame,capture_ready,automatic_ready,H,t0,blend\n";
      const auto until = GetTickCount64()+45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64()<until) step();
      require(observed.runtime && observed.renders && !observed.inject, "Mode1 actual effect failed to initialize without injected depth");
      check_unified_addon();
      query_frame = reinterpret_cast<frame_query_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"), "SunshineDepthTestFrame"));
      require(query_frame, "Mode1 fixture requires the existing passive current-frame adapter");
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_mode1);
      set_int("Depth_Map_View", 0); set_float("Depth_Adjustment", 100); set_float("Sharpen_Power", 0);
      find_texture("DoubleTex", exported, width*2, DXGI_FORMAT_R16G16B16A16_FLOAT);
      std::printf("DIAGNOSTIC mode1 trailing_clear=%d output=%ux%u depth=%ux%u native=%llu\n", preserved, width, height,
        scene->width, scene->height, static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(scene->texture.p)));
      settle("startup");
      require(scalar("Sunshine_CameraDepthScale")==8.f && zero()[1]==.125f,
        "Mode1 startup did not use the source's actual center mean");
      verify_depth();
      const auto lifetime = captured.source_id;
      for (unsigned pattern : {1u, 9u, 1u, 9u}) {
        scene->pattern = pattern;
        tick("changed-pattern");
        require(current() && captured.source_id==lifetime, "Mode1 pattern change lost the same current physical source");
        verify_depth();
      }
      std::puts("PASS lower-resolution mode1 current copies match four alternating spatial depth patterns at576 cells");

      // A depthless present must display its own new color, never the prior SBS.
      drawing = false;
      for (size_t i=0; i<size_t(width)*height; ++i) {
        const std::uint16_t red = 0x3a00;
        std::memcpy(source_bytes.data()+i*8, &red, 2);
      }
      fill_upload(source_upload, backbuffers[0]->GetDesc(), source_bytes.data(), source_footprint);
      for (unsigned i=0; i<4; ++i) {
        tick("missing-depth-new-color");
        require(!captured_ready && !ready() && scalar("Sunshine_CameraStrengthBlend")==0.f,
          "Mode1 missing depth reused a prior current copy or stereo state");
      }
      check_current_mono();
      drawing = true;
      settle("depth-return"); verify_depth();
      set_float("Depth_Adjustment", 0);
      for (unsigned i=0; i<4; ++i) tick("strength-zero");
      const auto mono_pixels = check_current_mono();
      set_float("Depth_Adjustment", 100);
      settle("stereo-restored");
      observed.capture = true; tick("physical-mono"); observed.capture = false;
      require(current() && ready() && read(mono.p, D3D12_RESOURCE_STATE_COPY_DEST)==source_bytes,
        "Mode1 stereo rendering modified the game's physical mono image");
      const auto stereo = read(exported.p);
      for (unsigned y=0; y<height; ++y) for (unsigned x=0; x<width*2; ++x) for (unsigned c=0; c<4; ++c)
        require(std::isfinite(channel(stereo,x,y,c)), "Mode1 actual stereo has nonfinite RGBA");
      require(image_difference(mono_pixels,stereo)>.002f, "Mode1 ready depth did not affect actual stereo pixels");
      sunshine_parity::write_bytes(runtime_directory/"mode1-final.sbs",stereo.data(),stereo.size());
      sunshine_parity::write_bytes(runtime_directory/"mode1-current-source.bin",source_bytes.data(),source_bytes.size());
      std::puts("PASS mode1 actual lower-resolution capture, current-color mono gap/recovery, finite stereo response and byte-exact physical mono");
      render_tracked_depth = {};
      reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_mode1);
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if (argc!=8) { std::fputs("usage: mode1_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> <fallback|preserved> width height\n",stderr); return 2; }
  std::thread([]{Sleep(120000);std::fputs("FAIL mode1 runtime watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    width=unsigned(std::stoul(argv[6])); height=unsigned(std::stoul(argv[7]));
    require(width>=640 && width<=3840 && height>=360 && height<=2160 && width%4==0 && height%4==0,"Invalid mode1 dimensions");
    const std::string kind=argv[5]; require(kind=="fallback" || kind=="preserved","Unknown mode1 case");
    require(sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") &&
      sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST") &&
      !sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST"),"Mode1 fixture needs Automatic/test adapter and mode2 flag unset");
    require(!fs::exists(fs::absolute(argv[4])),"Mode1 fixture requires fresh isolated output");
    mode1_fixture fixture; fixture.runtime_directory=fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),fixture.runtime_directory,2,0,fs::absolute(argv[3]));
    fixture.run(kind=="preserved"); return 0;
  } catch (const std::exception &error) { std::fprintf(stderr,"FAIL %s\n",error.what());return 1; }
}
