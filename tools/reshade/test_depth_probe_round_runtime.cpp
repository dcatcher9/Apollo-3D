// SPDX-License-Identifier: GPL-3.0-only
// Actual D3D12 challenger promotion through native Game 3D with no installed
// FX, captured at the end of the frame. No injected selection, depth,
// readiness or camera.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"
#include <atomic>

namespace {
  // Set once D3D12 destroys the resource it is attached to as private data
  // (the resource then releases its private data), the way ReShade itself
  // learns of a destroyed resource. Released on whichever thread drops the
  // resource's last reference.
  struct destruction_flag final : IUnknown {
    std::atomic<bool> &destroyed;
    std::atomic<ULONG> references{1};
    explicit destruction_flag(std::atomic<bool> &flag) : destroyed(flag) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **out) override {
      if (!out) return E_POINTER;
      if (riid == __uuidof(IUnknown)) { *out = static_cast<IUnknown *>(this); AddRef(); return S_OK; }
      *out = nullptr; return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
    ULONG STDMETHODCALLTYPE Release() override {
      const ULONG left = --references;
      if (!left) { destroyed.store(true); delete this; }
      return left;
    }
  };
  // {7C2E4D61-3B8A-4F0E-9A51-6D2C8B9E1F47}
  constexpr GUID destruction_flag_guid = {0x7c2e4d61, 0x3b8a, 0x4f0e, {0x9a, 0x51, 0x6d, 0x2c, 0x8b, 0x9e, 0x1f, 0x47}};

  struct probe_round_fixture : raw_runtime_fixture {
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
    // --set-challenger: native members drawn one per Present beside the
    // every-Present incumbent (the first set_size of them).
    std::array<std::unique_ptr<target_t>,3> native_set;
    unsigned set_size=3, set_present=0;
    target_t *set_rendered=nullptr;
    // --set-repromotion-kept: the first set, still allocated, never drawn again.
    std::array<std::unique_ptr<target_t>,3> abandoned_set;
    // --set-repromotion: set as D3D12 destroys each first-set member.
    std::array<std::atomic<bool>,3> first_set_destroyed{};
    // The capture owner's view of the first set (test add-on queries): the
    // id of the end-of-frame capture a source holds for its depth frame, and
    // whether a capture slot still holds a capture id (the query answers only
    // while one does).
    BOOL (*capture_ticket)(api::effect_runtime *,std::uint64_t,std::uint64_t *)=nullptr;
    BOOL (*capture_held)(std::uint64_t,BOOL *)=nullptr;
    // From the Present that dropped the first set (tick() counts): each
    // member's last capture copy and the Present on which its slot was
    // reclaimed, and the Present on which D3D12 destroyed the member.
    struct release_watch {
      bool active=false;
      unsigned presents=0;
      std::array<std::uint64_t,3> copy{};
      std::array<unsigned,3> copy_reclaimed{},destroyed{};
    } released_set;
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
      if (native_set[0]) {
        draw(*scene);
        set_rendered=native_set[set_present++%set_size].get();
        draw(*set_rendered);
        return;
      }
      if (rotating_lower) {
        const unsigned slot=lower_present++%3;
        lower_rendered=slot ? lower_peers[slot-1].get() : scene.get();
        draw(*lower_rendered);
        if (decoy) draw(*decoy); // New full-resolution challenger renders every present.
        return;
      }
      if (missing) {
        // Real useful raster depth with no legal end-of-frame capture. A
        // null/null aliasing barrier after its draw is legal without naming a
        // resource, and leaves no state of it proven (the matrix's D10), so its
        // copy is refused. Each later target's own clear proves it again.
        draw(*missing, false);
        D3D12_RESOURCE_BARRIER alias {};
        alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        commands->ResourceBarrier(1, &alias);
      }
      draw(*scene);
      if (decoy) draw(*decoy);
      for (const auto &flat : flats) draw(*flat);
    }
    void tick(const char *phase) {
      step(); game3d.no_effects();
      frame = game3d.last_render();
      require(frame.sequence > rendered_sequence, "A Present had no native Game 3D render to observe");
      rendered_sequence = frame.sequence;
      trace << phase << ',' << clock.now() << ',' << frame.sequence << ','
        << frame.depth.source_id << ',' << frame.depth.source_resource.handle << ','
        << frame.depth.frame_index << ',' << frame.depth.ready << ',' << frame.parameters.camera_ready << ','
        << frame.parameters.depth_scale << ',' << frame.parameters.strength_blend << '\n';
      require(trace.good(), "Cannot record challenger-promotion trajectory");
      if (released_set.active) {
        auto &watch=released_set;
        ++watch.presents;
        for (unsigned m=0;m<3;++m) {
          BOOL copy_out=FALSE;
          if (!watch.copy_reclaimed[m] && !capture_held(watch.copy[m],&copy_out)) watch.copy_reclaimed[m]=watch.presents;
          if (!watch.destroyed[m] && first_set_destroyed[m]) watch.destroyed[m]=watch.presents;
        }
      }
      if (missing)
        require(frame.depth.source_resource.handle != reinterpret_cast<std::uint64_t>(missing->texture.p),
                "A candidate without a legal current capture displaced useful scene depth");
    }
    void await_current(const char *phase, const target_t &target, unsigned limit_ms) {
      const auto started = clock.now();
      do { tick(phase); } while (!current(target) && clock.now() - started < limit_ms);
      std::printf("MEASURE %s promotion_ms=%llu limit_ms=%u source=%llu\n", phase,
        static_cast<unsigned long long>(clock.now() - started), limit_ms,
        static_cast<unsigned long long>(frame.depth.source_id));
      require(current(target) && clock.now() - started <= limit_ms,
              "Qualified native challenger did not promote within its bounded visit; inspect trajectory and ReShade log");
    }
    // The production Dump 3D capture of one native render: the exact depth
    // allocation it consumed, read through its own artifact.
    // A set's members draw the same content: the capture may be any of them.
    void verify_pattern(const char *phase, const target_t &expected, std::initializer_list<const target_t *> peers = {}) {
      const auto consumed = game3d.capture(phase, false, [&] { tick(phase); });
      const target_t *source = &expected;
      for (const auto *peer : peers)
        if (consumed.source_resource == reinterpret_cast<std::uint64_t>(peer->texture.p)) source = peer;
      const auto &target = *source;
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
      const auto until=clock.now()+milliseconds;
      do { tick(phase); require(current(target), "Useful selected source was lost during stable rendering"); }
      while (clock.now()<until);
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
      const auto began=clock.now(); unsigned consecutive=0;
      float scale=0.f; std::array<float,2> zero{};
      do {
        tick("rotating-lower-calibration");
        // A member without its own current copy yet (not qualified) may show
        // the previous Present's depth again, held (reused) within the source
        // age (the alternating fixture checks each hold); never another member
        // as this Present's own depth.
        require(!frame.depth.ready || frame.depth.reused_depth || current(*lower_rendered), "Rotating lower-res control admitted another present's member");
        if (current(*lower_rendered) && full()) {
          if (!consecutive) { scale=frame.parameters.depth_scale; zero=frame.parameters.convergence; }
          require(std::isfinite(scale) && scale>0.f, "Rotating lower-res control has no positive automatic scale");
          if (frame.parameters.depth_scale==scale && frame.parameters.convergence==zero) ++consecutive; else consecutive=0;
        } else consecutive=0;
      } while (consecutive<12 && clock.now()-began<15000);
      require(consecutive>=12, "Rotating lower-res sources did not calibrate before the challenger scheduling test");
      std::printf("MEASURE rotating-lower H=%.9g t0=%.9g\n", scale, zero[1]);
      const auto steady_until=clock.now()+1200;
      do {
        tick("rotating-lower-steady");
        require(current(*lower_rendered) && full() && frame.parameters.depth_scale==scale && frame.parameters.convergence==zero,
          "Calibrated rotating sources lost current readiness or their scale before introduction of the native challenger");
      } while (clock.now()<steady_until);
      std::puts("PASS calibrated lower-res ABC remains current and full-strength before challenger introduction");
      decoy=target(width,height,1,9);
      await_current("rotating-raw-does-not-starve-native",*decoy,4000);
      verify_pattern("rotating-raw-does-not-starve-native",*decoy);
      stable("promoted-native-survives-rotating-lower",*decoy,1200);
      verify_pattern("promoted-native-survives-rotating-lower",*decoy);
      std::puts("PASS per-member raw calibration sampling does not starve independent native-depth challenger qualification/promotion");
    }
    // check_rotating_lower_promotion's roles reversed: three native members
    // with identical useful content, one drawn per Present, challenge the
    // calibrated every-Present lower-resolution incumbent. Judged as one
    // capture set they cover every Present, so they promote on resolution;
    // judged one member at a time each would miss two Presents in three and
    // never promote. No Present may be without depth on the way (a member
    // without its own current copy yet holds the previous Present's depth).
    void check_set_challenger() {
      for (auto &member:native_set) member=target(width,height,1,9);
      const auto began=clock.now(); unsigned consecutive=0;
      do {
        tick("native-set-challenger");
        require(frame.depth.ready, "A Present had no depth while a native A/B/C set challenged every-Present lower-resolution depth");
        consecutive=current(*set_rendered) && !frame.depth.reused_depth && full() ? consecutive+1 : 0;
      } while (consecutive<30 && clock.now()-began<12000);
      std::printf("MEASURE native-set-challenger promotion_ms=%llu consecutive=%u\n",
        static_cast<unsigned long long>(clock.now()-began), consecutive);
      require(consecutive>=30, "A native A/B/C set covering every Present did not promote over lower-resolution depth");
      const auto steady_until=clock.now()+1200;
      do {
        tick("native-set-steady");
        require(current(*set_rendered) && !frame.depth.reused_depth && full(),
          "The promoted native set lost its own current depth on a Present");
      } while (clock.now()<steady_until);
      verify_pattern("native-set-promoted",*native_set[0],{native_set[1].get(),native_set[2].get()});
      std::puts("PASS a native A/B/C set judged as one capture set promotes over every-Present lower-resolution depth with no Present without depth");
    }
    // A better source that appears a second time. The promoted A/B/C set goes
    // away, the every-Present lower-resolution incumbent recovers, and a new
    // native A/B set then challenges it: it must promote within the budget a
    // set gets from a fresh start, with no Present without depth on the way,
    // and then keep its own current depth on every Present (no flapping back
    // to the incumbent it replaced). The first set is either destroyed, and
    // then the add-on must let D3D12 destroy it (hold no reference to a
    // buffer the game released), or kept allocated and never drawn again
    // (`keep_first_set`: an abandoned set never shares a capture set with a
    // working one).
    void check_set_repromotion(bool keep_first_set) {
      check_set_challenger();
      // Each member's end-of-frame capture copy, read on its own Present: a
      // capture slot that holds the member's depth frame (positive control).
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      capture_ticket=reinterpret_cast<decltype(capture_ticket)>(GetProcAddress(module,"SunshineDepthTestCaptureTicket"));
      capture_held=reinterpret_cast<decltype(capture_held)>(GetProcAddress(module,"SunshineStreamlineTestCaptureReadsCopyOut"));
      require(capture_ticket && capture_held,"The set release check needs the test add-on's capture queries");
      auto &watch=released_set;
      for (unsigned i=0;i<30 && std::count(watch.copy.begin(),watch.copy.end(),std::uint64_t{0});++i) {
        tick("native-set-copies");
        for (unsigned m=0;m<3;++m) {
          std::uint64_t id=0;BOOL copy_out=FALSE;
          if (set_rendered!=native_set[m].get() || !current(*set_rendered) || frame.depth.reused_depth) continue;
          require(capture_ticket(observed.runtime,reinterpret_cast<std::uint64_t>(set_rendered->texture.p),&id),
                  "The capture ticket query failed");
          if (id && capture_held(id,&copy_out)) watch.copy[m]=id;
        }
      }
      require(!std::count(watch.copy.begin(),watch.copy.end(),std::uint64_t{0}),
              "A promoted set member showed its own current depth without a held end-of-frame capture copy");
      auto &destroyed=first_set_destroyed;
      if (keep_first_set) abandoned_set=std::move(native_set);
      else for (unsigned i=0;i<3;++i) {
        auto *flag=new destruction_flag(destroyed[i]);
        require(SUCCEEDED(native_set[i]->texture->SetPrivateDataInterface(destruction_flag_guid,flag)),
          "Cannot watch the first set's destruction");
        flag->Release();
      }
      const auto released=clock.now();
      for (auto &member:native_set) member.reset();
      watch.active=true; // Present 1 is the first without the set.
      await_current(keep_first_set ? "set-abandoned-scaled-recovery" : "set-destroyed-scaled-recovery",*scene,12000);
      const unsigned recovered=watch.presents;
      stable("scaled-reconfirmed-after-set",*scene,1200);
      {
        // The capture owner reclaims each member's last capture copy whether
        // the game destroys the member or keeps it allocated: the copy's
        // ticket goes on the first depth frame its source did not work on
        // (Present 1), the copy and the backup copy read from it complete
        // with that Present's GPU work, which the fixture waits for, and the
        // next Present's capture poll reclaims the slot.
        constexpr unsigned copy_release_presents=4;
        while (std::count(watch.copy_reclaimed.begin(),watch.copy_reclaimed.end(),0u) && watch.presents<400)
          tick("set-copies-release");
        std::printf("MEASURE set-copies-release presents=%u,%u,%u limit=%u\n",watch.copy_reclaimed[0],watch.copy_reclaimed[1],
          watch.copy_reclaimed[2],copy_release_presents);
        for (const auto reclaimed:watch.copy_reclaimed)
          require(reclaimed && reclaimed<=copy_release_presents,
                  "The capture owner kept a dropped set member's end-of-frame capture copy (its ticket held the slot)");
        std::printf("PASS every dropped set member's capture copy is reclaimed within %u Presents\n",copy_release_presents);
      }
      if (!keep_first_set) {
        // The member views of a set that left the routed capture set retire
        // within 50 depth frames; nothing else may keep the game's buffers.
        constexpr unsigned release_limit_ms=4000;
        while (!(destroyed[0] && destroyed[1] && destroyed[2]) && clock.now()-released<release_limit_ms)
          tick("set-destroyed-release");
        std::printf("MEASURE set-destroyed-release release_ms=%llu limit_ms=%u destroyed=%d%d%d\n",
          static_cast<unsigned long long>(clock.now()-released),release_limit_ms,
          int(destroyed[0].load()),int(destroyed[1].load()),int(destroyed[2].load()));
        require(destroyed[0] && destroyed[1] && destroyed[2],
          "The add-on kept a depth-stencil the game released alive (its last end-of-frame capture held it)");
        // In Presents: the set leaves the routed capture set at the latest
        // when the incumbent recovered (its anchor then moves on), and its
        // member views retire 50 depth frames later (one per Present here),
        // which drops the add-on's last reference.
        constexpr unsigned retire_presents=50,slack_presents=4;
        std::printf("MEASURE set-destroyed-release presents=%u,%u,%u recovered=%u limit=%u\n",watch.destroyed[0],watch.destroyed[1],
          watch.destroyed[2],recovered,recovered+retire_presents+slack_presents);
        for (const auto present:watch.destroyed)
          require(present && present<=recovered+retire_presents+slack_presents,
                  "A destroyed set member outlived its capture views' retirement after the incumbent recovered");
        std::puts("PASS every member of a destroyed native set is released to D3D12 after the game drops it");
      }
      watch.active=false;
      set_size=2; set_present=0;
      for (unsigned i=0;i<set_size;++i) native_set[i]=target(width,height,1,9);
      std::printf("DIAGNOSTIC second-set originals incumbent=%llu A=%llu B=%llu\n",
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(scene->texture.p)),
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(native_set[0]->texture.p)),
        static_cast<unsigned long long>(reinterpret_cast<std::uint64_t>(native_set[1]->texture.p)));
      const auto began=clock.now(); unsigned consecutive=0;
      do {
        tick("second-set-challenger");
        require(frame.depth.ready, "A Present had no depth while a second native A/B set challenged the recovered incumbent");
        consecutive=current(*set_rendered) && !frame.depth.reused_depth && full() ? consecutive+1 : 0;
      } while (consecutive<30 && clock.now()-began<12000);
      std::printf("MEASURE second-set-challenger promotion_ms=%llu consecutive=%u\n",
        static_cast<unsigned long long>(clock.now()-began), consecutive);
      require(consecutive>=30, "A second native set covering every Present did not promote over the recovered lower-resolution incumbent");
      std::puts("PASS a native A/B set appearing after an A/B/C set came and went promotes over the recovered every-Present incumbent with no Present without depth");
      const auto steady_until=clock.now()+4000;
      do {
        tick("second-set-steady");
        require(current(*set_rendered) && !frame.depth.reused_depth && full(),
          "The re-promoted native set lost its own current depth on a Present (flapped back to the incumbent)");
      } while (clock.now()<steady_until);
      verify_pattern("second-set-promoted",*native_set[0],{native_set[1].get()});
      std::puts("PASS the re-promoted native A/B set keeps its own current depth on every Present for 4 s (no flapping)");
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
    void run(bool native_priority=false,bool set_challenger=false,bool set_repromotion=false,bool keep_first_set=false) {
      create_pipeline(); normal=false;
      scene=target(width/2,height/2,1,1);
      render_tracked_depth=[&]{render_depth();};
      trace.open(runtime_directory/"probe-round-trajectory.csv");
      trace << std::setprecision(17) << "phase,game_ms,render,source,resource,frame,depth_ready,camera_ready,H,blend\n";
      const auto until=clock.wall_now()+45000;
      while ((!observed.runtime || !observed.renders) && clock.wall_now()<until) step();
      require(observed.runtime && observed.renders && !observed.inject, "Actual probe-round fixture did not initialize");
      check_unified_addon();
      game3d.start();
      game3d.set_strength(100);
      rendered_sequence=game3d.await_render([&]{ step(); });
      // Setup is done: game time and the trace from here on.
      game3d.begin_game_time();
      await_current("sole-scaled-startup",*scene,12000);
      const auto calibrated=clock.now()+12000;
      while (!(current(*scene) && full()) && clock.now()<calibrated)
        tick("sole-scaled-calibration");
      require(current(*scene) && full(),
              "Incumbent must be genuinely calibrated before testing challenger scheduling");
      stable("sole-scaled-confirmed",*scene,1200);

      if (native_priority) {
        check_native_priority();
        render_tracked_depth={};
        return;
      }
      if (set_challenger) {
        check_set_challenger();
        render_tracked_depth={};
        return;
      }
      if (set_repromotion) {
        check_set_repromotion(keep_first_set);
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
      check_rotating_lower_promotion();
      render_tracked_depth={};
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  sunshine_game3d_fixture::take_wall_clock_option(argc,argv);
  const char *mode=argc==6 ? argv[5] : "";
  if(argc!=5 && (argc!=6 || (std::strcmp(mode,"--native-priority")!=0 && std::strcmp(mode,"--set-challenger")!=0 &&
                              std::strcmp(mode,"--set-repromotion")!=0 && std::strcmp(mode,"--set-repromotion-kept")!=0))){
    std::fputs("usage: depth_probe_round_runtime <official.dll> <frozenShaders> <test.addon64> <fresh-output> "
               "[--native-priority|--set-challenger|--set-repromotion|--set-repromotion-kept] [--wall-clock]\n",stderr);return 2;
  }
  std::thread([]{Sleep(150000);std::fputs("FAIL probe-round watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    select_native_boot();
    width=3840; height=2160;
    require(!fs::exists(fs::absolute(argv[4])),"Fresh probe-round output required");
    probe_round_fixture f; f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));
    const bool kept=std::strcmp(mode,"--set-repromotion-kept")==0;
    f.run(std::strcmp(mode,"--native-priority")==0,std::strcmp(mode,"--set-challenger")==0,
          kept || std::strcmp(mode,"--set-repromotion")==0,kept); f.clock.report("probe-round"); return 0;
  } catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
