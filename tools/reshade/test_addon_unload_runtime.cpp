// SPDX-License-Identifier: GPL-3.0-only
// The add-on's lifecycle under ReShade, the session-level tests' third kind
// (README, Session-level tests). ReShade unloads its add-ons whenever a game
// destroys its last device and loads them again for the next one; the add-on
// pins its module, so ReShade initializes the same mapped module again
// (docs/reshade-sbs.md, the add-on session). Assassin's Creed Shadows creates
// and destroys a D3D12 device at startup while module discovery and a shader
// compile the add-on had started on pool threads still ran its code: the game
// crashed executing the unmapped add-on or D3DCompiler (fixed by 5148a6da).
//
// One process and one hidden window, with the official runtime, the test
// add-on, the 2.7.30 interposer and the NGX stand-in (the session timelines'
// fixture, test_game3d_session_timeline_runtime.cpp), through these devices:
// 1. a startup device with no queue, swapchain or frame, destroyed while its
//    initialization's pool work (module discovery, the content sampler's
//    compile) runs;
// 2. the game's first device, through the FX boot and Streamline discovery
//    until Game 3D renders, then destroyed;
// 3. a device of another swapchain size, destroyed while its shader entries
//    compile on the pool;
// 4. a device destroyed while its renderer builds on the pool (the renderer's
//    build hooks delay each step);
// 5. three game devices in turn, each verified by a session timeline, then
//    destroyed with frames in flight: the add-on's copies of the last frames
//    still unobserved, DLSS-G's compute copies whose ring entries were never
//    reset, and a last frame recorded and submitted but never presented,
//    whose command-less FG depth tag and UI tags leave copies deferred to that
//    Present.
// Every device also leaves one held pool callback of the add-on's own code,
// and, as ReShade destroys it, lines queued in the add-on's asynchronous log
// writer.
//
// Each unload must leave the module mapped; the held callback must then
// compile and return, the queued lines must be written in order, the shader
// compiles and renderer builds the add-on started must finish or be released,
// and the device must end: the next native device carries no private data of
// the one before. Game 3D must render on each game device, reloaded after the
// unloads before it, and each game device's session timeline (its continuity
// invariant and oracle) must render Game 3D from API depth, an FG scope's
// depth and the game's UI pair offered to detection.
//
// Known gap L1 (lifecycle_gap): a device whose session captured depth
// outlives the game's release of it. ReShade's ID3D12Resource::GetDevice hook
// returns its device proxy with a reference, and the add-on's capture storage
// and source references keep those, which only a later Present's storage
// reuse or retirement releases. Once the game stops presenting and releases
// the device, ReShade therefore never destroys it or unloads the add-on (no
// destroy_device event comes), and the game's next D3D12CreateDevice
// returns the same device; there Game 3D must render depth, FG depth and UI
// again, and a further cycle must leave the process's handles, threads and
// private bytes within a bound. The capture registry also keeps each released
// queue whose captures it retains, one more each cycle, and it holds four:
// the device's fourth session (its third recreation) lost FG depth here
// (mono, FG_scope_required), so the test makes two recreations. Each game device of item 5 witnesses the gap; a run in which
// one is destroyed fails (the gap closed: remove its row and require the
// unload and reload of the in-flight devices instead).
//
// Usage: reshade_addon_unload_runtime_test official.dll frozenShaders
// test.addon64 freshOutput frame_generation_interposer.dll
#define SUNSHINE_SESSION_TIMELINE_FIXTURE_ONLY
// The fixtures define their Windows macros before any standard header.
// clang-format off
#include "test_game3d_session_timeline_runtime.cpp"
// clang-format on
#include <psapi.h>
#include <tlhelp32.h>

namespace {
  constexpr wchar_t addon_file[] = L"SunshineSBSTest.addon64";

  // ReShade's interface ID that returns a proxy's original object.
  constexpr GUID unwrapped_object {0x7f2c9a11, 0x3b4e, 0x4d6a, {0x81, 0x2f, 0x5e, 0x9c, 0xd3, 0x7a, 0x1b, 0x42}};
  // Private data this fixture sets on each native device it creates.
  constexpr GUID device_marker {0x3c9b6d52, 0x8e41, 0x4f0a, {0x9d, 0x27, 0x61, 0xb8, 0x0e, 0x5a, 0xc3, 0x14}};

  // The lifecycle's known gaps, under the matrix's known-gap discipline:
  // each names the limitation its witnesses meet; a witness that does not
  // meet it fails the run (the gap closed, so its fix removes the row).
  struct lifecycle_gap {
    const char *id, *limitation;
  };

  constexpr lifecycle_gap captured_device_gap {"L1-captured-device-outlives-release", "a device whose session captured depth outlives the game's release (the add-on's capture storage keeps ReShade's device proxy)"};

  // A verified device's timeline: the viewport's SR depth with the game's UI
  // pair, an FG scope's depth tags, DLSS-G's compute evaluations, then SR
  // depth again with the pair on the game's list.
  constexpr char verify_events[] = "30*sr=local,ui=bare; fg=on,tag=present_list,ui=bare; 30*tag=present_list,ui=bare; 10*dlssg=compute,ui=bare; fg=off,sr=local; 30*sr=local,ui=listed";
  // The frames a destroyed game device leaves in flight: SR depth, then FG
  // with DLSS-G's compute evaluations (three ring entries, none reset again),
  // and a last frame that the game records and submits but never presents: a
  // command-less FG depth tag, whose copy is deferred to that Present, the
  // game's command-less UI tags, deferred likewise, its SR evaluation and
  // DLSS-G's compute evaluation.
  constexpr char in_flight_events[] = "20*sr=local,ui=bare; fg=on,dlssg=compute,ui=bare; 5*sr=local,dlssg=compute,ui=bare; tag=present_bare,sr=local,dlssg=compute,ui=bare";
  // The game's one Streamline viewport, which it keeps across its devices as
  // games do. (A new viewport's FG scope shows no depth while an earlier one
  // that used FG is still among the provider's viewports, which the
  // timelines' isolation retires, so the stretches here never change it.)
  constexpr std::uint32_t game_viewport = 4000;
  constexpr unsigned queued_lines = 128;

  // ReShade's destroy_device event, before ReShade unloads its add-ons:
  // queues lines in the add-on's asynchronous writer, so that it is still
  // writing them while the add-on uninitializes.
  struct teardown_probe_t {
    BOOL (*log)(const char *, unsigned) {};
    unsigned (*backlog)(BOOL *) {};
    std::string prefix;
    bool armed {};
    unsigned destroyed {}, backlog_at_destroy {};
  } teardown_probe;

  void probe_destroy_device(api::device *) {
    ++teardown_probe.destroyed;
    if (!teardown_probe.armed) {
      return;
    }
    teardown_probe.armed = false;
    teardown_probe.log(teardown_probe.prefix.c_str(), queued_lines);
    teardown_probe.backlog_at_destroy = teardown_probe.backlog(nullptr);
  }

  // What each Present of a verified device showed, from its frame record and
  // its native render.
  struct present_tally_t {
    BOOL (*last_render)(api::effect_runtime *, sunshine_game3d::test::last_render *) {};
    unsigned presents {}, rendered {}, api_depth {}, fg_depth {}, ui_pair {};
  } present_tally;

  void tally_present(api::effect_runtime *runtime) {
    using sunshine_game3d::source_api;
    using sunshine_game3d::source_slot;
    using sunshine_game3d::source_state;
    ++present_tally.presents;
    sunshine_game3d::frame_record r;
    if (record_history.query && record_history.query(runtime, &r) && r.present) {
      const auto &owner = r[r.depth_owner == source_slot::none ? source_slot::hold : r.depth_owner];
      const bool api_owner = r.depth_owner == source_slot::fg_depth || r.depth_owner == source_slot::scene_depth;
      if (api_owner && (owner.state == source_state::ready || owner.state == source_state::held) && (owner.key.api == source_api::streamline || owner.key.api == source_api::ngx)) {
        ++present_tally.api_depth;
        present_tally.fg_depth += r.depth_owner == source_slot::fg_depth;
      }
      const auto &hudless = r[source_slot::hudless], &backbuffer = r[source_slot::backbuffer];
      present_tally.ui_pair += hudless.state == source_state::ready && backbuffer.state == source_state::ready && hudless.key.api == source_api::streamline;
    }
    sunshine_game3d::test::last_render render;
    if (present_tally.last_render && present_tally.last_render(runtime, &render) && render.rendered && render.parameters.depth_ready) {
      ++present_tally.rendered;
    }
  }

  // The process's resources at the end of a device.
  struct process_usage {
    DWORD handles {}, threads {}, gdi {}, user {};
    std::uint64_t private_bytes {};

    static process_usage now() {
      process_usage out;
      const auto self = GetCurrentProcess();
      GetProcessHandleCount(self, &out.handles);
      out.gdi = GetGuiResources(self, GR_GDIOBJECTS);
      out.user = GetGuiResources(self, GR_USEROBJECTS);
      PROCESS_MEMORY_COUNTERS_EX memory {};
      memory.cb = sizeof(memory);
      if (GetProcessMemoryInfo(self, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&memory), sizeof(memory))) {
        out.private_bytes = memory.PrivateUsage;
      }
      const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
      if (snapshot != INVALID_HANDLE_VALUE) {
        THREADENTRY32 entry {};
        entry.dwSize = sizeof(entry);
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
          out.threads += entry.th32OwnerProcessID == GetCurrentProcessId();
        }
        CloseHandle(snapshot);
      }
      return out;
    }

    void print(const char *label) const {
      std::printf("MEASURE addon-lifecycle usage %s handles=%lu threads=%lu gdi=%lu user=%lu private_kib=%llu\n", label, static_cast<unsigned long>(handles), static_cast<unsigned long>(threads), static_cast<unsigned long>(gdi), static_cast<unsigned long>(user), static_cast<unsigned long long>(private_bytes / 1024));
    }

    // Whether later grew beyond the pool's idle threads and allocator slack.
    static bool bounded(const process_usage &earlier, const process_usage &later) {
      const auto within = [](std::uint64_t after, std::uint64_t before, std::uint64_t bound) {
        return after <= before + bound;
      };
      return within(later.handles, earlier.handles, 64) && within(later.threads, earlier.threads, 8) && within(later.gdi, earlier.gdi, 4) && within(later.user, earlier.user, 4) && within(later.private_bytes, earlier.private_bytes, 32ull << 20);
    }
  };

  struct lifecycle_fixture: timeline_fixture {
    // The add-on's module, from the startup device on.
    HMODULE pinned {};
    unsigned registrations {}, unloads {}, devices {}, witnessed {};
    // The current device's session captured depth; the last device ended
    // alive (known gap L1).
    bool captured {}, survived {};
    unsigned (*shader_entries_pending)(unsigned *) {};
    void (*renderer_builds)(unsigned *, unsigned *) {};
    void (*renderer_step_delay)(unsigned) {};
    BOOL (*set_diagnostics)(BOOL) {};
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) ngx_release {};

    // -- The add-on across ReShade's unload ---------------------------------

    void require_registered(const char *label) {
      ++registrations;
      if (!pinned) {
        pinned = GetModuleHandleW(addon_file);
      }
      if (reshade_log_count({"Registered add-on \"Sunshine SBS TEST ONLY\""}) == registrations && pinned && GetModuleHandleW(addon_file) == pinned) {
        return;
      }
      throw std::runtime_error(std::string("ReShade did not initialize the same pinned add-on module for the ") + label);
    }

    void require_unloaded(const char *label) {
      ++unloads;
      if (reshade_log_count({"Unloading add-on \"Sunshine SBS TEST ONLY\""}) != unloads || reshade_log_count({"Unregistered add-on \"Sunshine SBS TEST ONLY\""}) != unloads) {
        throw std::runtime_error(std::string("ReShade did not uninitialize and unload the add-on with the ") + label);
      }
      if (GetModuleHandleW(addon_file) != pinned) {
        throw std::runtime_error(std::string("ReShade's unload with the ") + label + " unmapped the add-on; AddonInit must pin the module");
      }
    }

    // The add-on's lifecycle observations and the fixture's adapters, bound
    // once the add-on is loaded; the pinned module keeps them.
    void bind_exports() {
      shader_entries_pending = reinterpret_cast<decltype(shader_entries_pending)>(GetProcAddress(pinned, "SunshineGame3DTestShaderEntriesPending"));
      renderer_builds = reinterpret_cast<decltype(renderer_builds)>(GetProcAddress(pinned, "SunshineGame3DTestRendererBuilds"));
      renderer_step_delay = reinterpret_cast<decltype(renderer_step_delay)>(GetProcAddress(pinned, "SunshineGame3DTestRendererStepDelay"));
      set_diagnostics = reinterpret_cast<decltype(set_diagnostics)>(GetProcAddress(pinned, "SunshineGame3DTestSetDiagnostics"));
      teardown_probe.log = reinterpret_cast<decltype(teardown_probe.log)>(GetProcAddress(pinned, "SunshineSbsTestLog"));
      teardown_probe.backlog = reinterpret_cast<decltype(teardown_probe.backlog)>(GetProcAddress(pinned, "SunshineSbsTestLogBacklog"));
      present_tally.last_render = reinterpret_cast<decltype(present_tally.last_render)>(GetProcAddress(pinned, "SunshineGame3DTestLastRender"));
      ngx_release = reinterpret_cast<decltype(ngx_release)>(GetProcAddress(GetModuleHandleW(nullptr), "NVSDK_NGX_D3D12_ReleaseFeature"));
      require(shader_entries_pending && renderer_builds && renderer_step_delay && set_diagnostics && teardown_probe.log && teardown_probe.backlog && present_tally.last_render && ngx_release, "The test add-on lacks its lifecycle observations, or the fixture its NGX release");
      reshade::register_event<reshade::addon_event::destroy_device>(probe_destroy_device);
      reshade::register_event<reshade::addon_event::reshade_present>(tally_present);
    }

    // One pool callback of the add-on's own code, held until after an
    // unload, then compiling through the add-on's D3DCompiler import.
    struct held_work {
      HANDLE proceed {}, returned {};
      volatile LONG compiled {};
      std::uint64_t released {};
    };

    std::unique_ptr<held_work> hold_pool_work() {
      using hold_t = BOOL (*)(HANDLE, HANDLE, volatile LONG *);
      const auto hold = reinterpret_cast<hold_t>(GetProcAddress(pinned, "SunshineSbsTestHoldPoolWork"));
      auto work = std::make_unique<held_work>();
      work->proceed = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      work->returned = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      require(hold && work->proceed && work->returned && hold(work->proceed, work->returned, &work->compiled), "Could not start held pool work in the add-on");
      return work;
    }

    void release_held(held_work &work, const char *label) {
      SetEvent(work.proceed);
      const bool returned = WaitForSingleObject(work.returned, 30000) == WAIT_OBJECT_0;
      const auto compiled = InterlockedCompareExchange(&work.compiled, 0, 0);
      CloseHandle(work.proceed);
      CloseHandle(work.returned);
      if (!returned || compiled != 1) {
        throw std::runtime_error(std::string("The add-on's held pool work did not compile and return after the ") + label + " ended");
      }
      std::printf("MEASURE addon-lifecycle %s end_to_pool_return_ms=%llu\n", label, static_cast<unsigned long long>(GetTickCount64() - work.released));
    }

    // The lines the teardown probe queued, written in order once the add-on
    // was unloaded. Nothing presents meanwhile: no device is alive.
    void require_lines_written(const char *label) {
      const auto last = teardown_probe.prefix + " " + std::to_string(queued_lines) + " of " + std::to_string(queued_lines);
      for (const auto until = GetTickCount64() + 15000; !reshade_log_count({last.c_str()}) && GetTickCount64() < until;) {
        Sleep(5);
      }
      const auto lines = reshade_log_lines({teardown_probe.prefix.c_str()});
      bool ordered = lines.size() == queued_lines;
      for (unsigned i = 0; ordered && i != queued_lines; ++i) {
        ordered = lines[i].find(teardown_probe.prefix + " " + std::to_string(i + 1) + " of ") != std::string::npos;
      }
      std::printf("MEASURE addon-lifecycle %s queued_lines=%u backlog_at_destroy=%u written=%zu\n", label, queued_lines, teardown_probe.backlog_at_destroy, lines.size());
      if (!ordered || !teardown_probe.backlog_at_destroy) {
        throw std::runtime_error(std::string("The add-on's log writer lost, reordered or had already written the lines queued as the ") + label + " was destroyed");
      }
    }

    // After an unload: the held pool work completes, the queued lines are
    // written, and the shader compiles and renderer builds the add-on started
    // finish or are released.
    void after_unload(held_work &work, const char *label) {
      work.released = GetTickCount64();
      release_held(work, label);
      require_lines_written(label);
      unsigned blobs {}, started {}, in_flight {};
      for (const auto until = GetTickCount64() + 20000; shader_entries_pending(&blobs) && GetTickCount64() < until;) {
        Sleep(5);
      }
      renderer_builds(&started, &in_flight);
      std::printf("MEASURE addon-lifecycle %s shader_entries_pending=%u blobs=%u renderer_builds=%u in_flight=%u\n", label, shader_entries_pending(nullptr), blobs, started, in_flight);
      require(!shader_entries_pending(nullptr) && !in_flight, "Shader entries or a renderer build the add-on started stayed on the pool after the unload");
    }

    // -- The game's devices -------------------------------------------------

    // A new native device carries no marker of an earlier one (D3D12 returns
    // the adapter's existing device while one is alive), then gets its own.
    // After a device that outlived its release (known gap L1) the game gets
    // that device back, and ReShade does not initialize the add-on again.
    void mark_device(ID3D12Device *proxy, const char *label) {
      com_ptr<ID3D12Device> native;
      checked(proxy->QueryInterface(unwrapped_object, reinterpret_cast<void **>(native.put())), "Unwrap the game's device");
      unsigned earlier {};
      UINT size = sizeof(earlier);
      const bool marked = SUCCEEDED(native->GetPrivateData(device_marker, &size, &earlier));
      if (survived) {
        if (!marked || earlier != devices || reshade_log_count({"Registered add-on \"Sunshine SBS TEST ONLY\""}) != registrations) {
          throw std::runtime_error(std::string("The ") + label + " is neither the device that outlived its release (known gap L1) nor a new one ReShade initialized the add-on for");
        }
        std::printf("MEASURE addon-lifecycle %s gap=%s device=%u (the device the game released)\n", label, captured_device_gap.id, earlier);
        return;
      }
      if (marked) {
        char text[200];
        std::snprintf(text, sizeof(text), "The %s is device %u, which survived its destruction: a reference to that native device leaked", label, earlier);
        throw std::runtime_error(text);
      }
      require_registered(label);
      const unsigned id = ++devices;
      checked(native->SetPrivateData(device_marker, sizeof(id), &id), "Mark the game's native device");
    }

    void create_device(const char *label) {
      create_game_device();
      mark_device(game.p, label);
    }

    // The earlier session's records and provider lines; the next session's
    // Presents are numbered and logged anew. Called once the earlier
    // session's lines were written, before the new device's first Present.
    void forget_session_records() {
      record_history.records.clear();
      matched_present = 0;
      matched_lines = provider_lines().size();
    }

    // Game 3D renders on the current device.
    void require_renders(const char *label) {
      const auto awaited = GetTickCount64();
      sunshine_game3d::test::last_render render;
      for (const auto until = awaited + 45000; !(observed.runtime && present_tally.last_render(observed.runtime, &render) && render.sequence && render.rendered) && GetTickCount64() < until;) {
        present_frame();
      }
      if (!render.rendered) {
        throw std::runtime_error(std::string("Native Game 3D never rendered on the ") + label);
      }
      std::printf("MEASURE addon-lifecycle %s first_render_ms=%llu\n", label, static_cast<unsigned long long>(GetTickCount64() - awaited));
    }

    // A game device set up as the first was (load_native, setup): the
    // runtime with no FX, the Diagnostics switch, the camera, depth producer,
    // decoy and UI images, and the other queues; Game 3D renders.
    void game_device(const char *label) {
      create_device(label);
      forget_session_records();
      for (const auto until = GetTickCount64() + 30000; !observed.runtime && GetTickCount64() < until;) {
        step();
      }
      require(observed.runtime != nullptr, "The official runtime did not initialize the game's new device");
      check_unified_addon();
      require(set_diagnostics(diagnostics_off() ? FALSE : TRUE), "Test add-on cannot set the Diagnostics switch");
      initialize_camera();
      create_pipeline();
      create_compute();
      first = target_uav(width / 2, height / 2, 0);
      selected = first.get();
      active_area = {0, 0, selected->width, selected->height};
      decoy = target(width, height, 1, 9, false, true);
      set_foreground(window);
      setup_device();
      require_renders(label);
    }

    // A game ends its device: its queues finish, its NGX feature is released,
    // then its objects, swapchain and device go. ReShade destroys the
    // runtime, then the device, and with its last device unloads its
    // add-ons; returns whether it did. A device whose session captured depth
    // must outlive the release instead (known gap L1), and any other must not.
    bool end_device(const char *label) {
      render_tracked_depth = {};
      checked(queue->Signal(completion.p, ++fence_value), "Signal the presenting queue's last work");
      checked(completion->SetEventOnCompletion(fence_value, nullptr), "Wait for the presenting queue");
      if (game_queue.p) {
        checked(game_queue->Signal(game_done.p, ++game_value), "Signal the game queue's last work");
        checked(game_done->SetEventOnCompletion(game_value, nullptr), "Wait for the game queue");
      }
      if (compute_queue.p) {
        checked(compute_queue->Signal(compute_done.p, ++compute_value), "Signal DLSS-G's last work");
        checked(compute_done->SetEventOnCompletion(compute_value, nullptr), "Wait for DLSS-G's queue");
      }
      if (feature) {
        require(ngx_release(feature) == ngx_fixture::success, "The game's DLSS-G feature release failed");
        feature = nullptr;
      }
      dlssg_inputs = {};
      hudless.reset();
      backbuffer.reset();
      for (unsigned n = 0; n != game_ring; ++n) {
        game_lists[n].reset();
        game_allocators[n].reset();
      }
      for (unsigned n = 0; n != compute_ring; ++n) {
        compute_lists[n].reset();
        compute_allocators[n].reset();
      }
      game_queue.reset();
      game_done.reset();
      compute_queue.reset();
      compute_done.reset();
      game_values = {};
      compute_values = {};
      game_value = compute_value = 0;
      game_at = compute_at = 0;
      first.reset();
      second.reset();
      barrier_scratch.reset();
      selected = nullptr;
      decoy.reset();
      scene.reset();
      shadow.reset();
      compute_root.reset();
      compute_pipeline.reset();
      batch_queue.reset();
      root.reset();
      depth32_pipeline.reset();
      depth32s8_pipeline.reset();
      // Left: the device, the presenting queue and the swapchain (each of
      // which holds the device once); any other reference is another owner's.
      game->AddRef();
      const auto others = game->Release() - 3;
      const auto destroyed_before = teardown_probe.destroyed;
      teardown_probe.prefix = std::string("Sunshine lifecycle test: the ") + label + " line";
      teardown_probe.armed = true;
      release_game_device();
      observed.runtime = nullptr;
      game_native_command = 0;
      const bool destroyed = teardown_probe.destroyed > destroyed_before;
      teardown_probe.armed = false;
      std::printf("MEASURE addon-lifecycle %s ended destroyed=%u captured=%u other_device_references=%lu\n", label, unsigned(destroyed), unsigned(captured), static_cast<unsigned long>(others));
      const bool had_captured = captured;
      captured = false;
      survived = !destroyed;
      if (destroyed) {
        require_unloaded(label);
        if (had_captured) {
          throw std::runtime_error(std::string("Known gap ") + captured_device_gap.id + " closed: the " + label + " was destroyed after its session captured depth; remove the row and require the unload and reload of the in-flight devices");
        }
        return true;
      }
      if (!had_captured) {
        throw std::runtime_error(std::string("The ") + label + ", whose session captured nothing, outlived the game's release: a reference to it leaked");
      }
      ++witnessed;
      // ReShade names each released queue proxy whose original another
      // owner still references (the capture registry's retained queues).
      std::printf("MEASURE addon-lifecycle %s gap=%s other_device_references=%lu queues_still_referenced=%zu: %s\n", label, captured_device_gap.id, static_cast<unsigned long>(others), reshade_log_count({"Reference count for ID3D12CommandQueue", "is inconsistent"}), captured_device_gap.limitation);
      return false;
    }

    // -- Verification and work in flight ------------------------------------

    void begin_stretch(const timeline_case &value, const std::string &label) {
      result = {};
      frames = &value.frames;
      view = {{nullptr, sunshine_streamline::viewport_guid, 1}, game_viewport};
      last_capture = 0;
      api_supplied = other_queue_copy = fg_copy = fg_tag_copy = dlssg_copy = dlssg_evaluated = 0;
      present_frames.clear();
      where = label;
      continuity_log.judge = {};
      continuity_log.pending = {};
      continuity_log.where = where.c_str();
      script_name = where.c_str();
      script_deadline = GetTickCount64() + 15000 + 150 * value.frames.size();
    }

    void end_stretch() {
      script_name = nullptr;
      script_deadline = 0;
      continuity_log.where = "session";
    }

    static timeline_case event_list(const char *text) {
      timeline_case value;
      std::string error;
      require(parse_events(text, value.frames, error), error.c_str());
      return value;
    }

    // A session timeline (its continuity invariant and oracle) whose Presents
    // render Game 3D from API depth, show an FG scope's depth and offer the
    // game's UI pair to detection.
    void verify(const char *label) {
      if (fg_on) {
        // A game applies its options again to its new device.
        set_frame_generation(false);
      }
      captured = true;
      const auto value = event_list(verify_events);
      present_tally = {present_tally.last_render};
      next_viewport = game_viewport;
      const auto outcome = run(value, std::string(label) + "-verify");
      const auto tally = present_tally;
      std::printf("MEASURE addon-lifecycle %s verify frames=%u presents=%u legal=%u shown=%u failures=%zu rendered=%u api_depth=%u fg_depth=%u ui_pair=%u elapsed_ms=%llu", label, outcome.frames, outcome.presents, outcome.legal, outcome.shown, outcome.failures.size(), tally.rendered, tally.api_depth, tally.fg_depth, tally.ui_pair, static_cast<unsigned long long>(outcome.elapsed_ms));
      for (std::size_t r = 0; r != outcome.rows.size(); ++r) {
        if (outcome.rows[r].exercised) {
          std::printf(" %s={supplied=%u violations=%u}", continuity::known_gaps[r].id, outcome.rows[r].exercised, outcome.rows[r].violations);
        }
      }
      std::printf("\n");
      for (std::size_t f = 0; f != outcome.failures.size() && f != 8; ++f) {
        std::printf("MEASURE addon-lifecycle %s failure %s: %s\n", label, outcome.failures[f].kind.c_str(), outcome.failures[f].text.c_str());
      }
      if (!outcome.failures.empty()) {
        throw std::runtime_error(std::string("The ") + label + "'s session timeline failed: " + outcome.failures.front().text);
      }
      if (tally.rendered < 30 || tally.api_depth < 30 || tally.fg_depth < 5 || tally.ui_pair < 5) {
        throw std::runtime_error(std::string("Game 3D did not render the ") + label + "'s API depth, FG depth and UI pair");
      }
      std::printf("PASS addon-lifecycle %s: Game 3D renders API depth, an FG scope's depth and the game's UI pair, judged by a session timeline\n", label);
    }

    // The in-flight frames, then the last frame recorded and submitted with
    // its Present never made.
    void leave_frames_in_flight(const char *label) {
      captured = true;
      const auto value = event_list(in_flight_events);
      begin_stretch(value, std::string(label) + "-in-flight");
      for (frame = 0; frame + 1 != value.frames.size(); ++frame) {
        run_frame();
      }
      using namespace frame_events;
      const auto &last = value.frames[frame];
      require(of<fg_event>(last, events::fg) == fg_event::keep && !last[events::drs], "The unpresented frame changes no option");
      supply = {};
      decisions.clear();
      polled.clear();
      refusal_serial = last_refusal().serial;
      begin_commands();
      frame_body();
      submit(false);
      end_stretch();
      require(fg_on && deferred_tag, "The oracle names no copy deferred to the unpresented frame's Present");
      std::printf("MEASURE addon-lifecycle %s in-flight frames=%zu presents=%u deferred_tag=1 compute_entries=%u\n", label, value.frames.size(), continuity_log.judge.presents(), compute_ring);
    }
  };

  // -- The phases -------------------------------------------------------------

  // ReShade loads the add-on for a startup device with no queue, swapchain or
  // frame and unloads it with that device, while its initialization's pool
  // work and one held callback still run.
  void startup_device(lifecycle_fixture &fixture) {
    const auto create = reinterpret_cast<decltype(&D3D12CreateDevice)>(GetProcAddress(fixture.runtime_module, "D3D12CreateDevice"));
    require(create, "Official runtime lacks the D3D12 proxy export");
    require(!GetModuleHandleW(addon_file), "The add-on was loaded before any device existed");
    std::unique_ptr<lifecycle_fixture::held_work> work;
    {
      com_ptr<ID3D12Device> device;
      checked(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())), "Create the short-lived startup device");
      fixture.mark_device(device.p, "startup device");
      fixture.bind_exports();
      // Its initialization already started the add-on's module discovery and
      // content-sampler compile on pool threads; this one is held until after
      // the unload, so it is certainly still to run.
      work = fixture.hold_pool_work();
      teardown_probe.prefix = "Sunshine lifecycle test: the startup device line";
      teardown_probe.armed = true;
    }
    require(!teardown_probe.armed, "ReShade did not destroy the startup device with its last reference");
    fixture.require_unloaded("startup device");
    fixture.after_unload(*work, "startup device");
    std::puts("PASS the add-on stays mapped across ReShade's unload with its startup device, and its held pool compile completes");
  }

  // A device presented until pending holds, then destroyed with that work in
  // flight; nothing of the game's session is set up on it.
  template<class Pending>
  void brief_device(lifecycle_fixture &fixture, const char *label, const Pending &pending) {
    fixture.create_device(label);
    unsigned presents {};
    for (const auto until = GetTickCount64() + 30000; !(observed.runtime && pending()) && GetTickCount64() < until; ++presents) {
      fixture.step();
    }
    require(pending(), (std::string("The ") + label + " never had its work in flight").c_str());
    std::printf("MEASURE addon-lifecycle %s presents_before_destruction=%u\n", label, presents);
    auto work = fixture.hold_pool_work();
    require(fixture.end_device(label), "A brief device outlived the game's release");
    fixture.after_unload(*work, label);
  }
}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 6) {
    std::fputs("usage: reshade_addon_unload_runtime_test <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory> <frame_generation_interposer.dll>\n", stderr);
    return 2;
  }
  std::thread([] {
    Sleep(170000);
    std::fputs("FAIL add-on lifecycle runtime watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    select_native_boot();
    require(_putenv_s("SUNSHINE_DEPTH_GENERIC_ONLY_TEST", "0") == 0, "Could not configure the lifecycle runtime");
    width = 1280;
    height = 720;
    const auto directory = fs::absolute(argv[4]);
    require(!fs::exists(directory), "The add-on lifecycle fixture requires a fresh isolated output directory");
    fs::create_directories(directory / "local-app-data");
    // The add-on's per-user shader store, fresh: each swapchain size compiles.
    require(SetEnvironmentVariableW(L"LOCALAPPDATA", (directory / "local-app-data").c_str()), "Could not isolate the add-on's shader store");
    const auto interposer = fs::absolute(argv[5]);
    require(fs::is_regular_file(interposer), "The 2.7.30 interposer DLL is missing");
    const auto isolated = directory / "sl.interposer.dll";
    fs::copy_file(interposer, isolated);
    const auto sdk = LoadLibraryW(isolated.c_str());
    require(sdk, "Could not load the isolated interposer");
    lifecycle_fixture fixture;
    fixture.runtime_directory = directory;
    fixture.before_game_device = [&] {
      startup_device(fixture);
    };
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), directory, 2, 0, fs::absolute(argv[3]));
    fixture.mark_device(fixture.game.p, "first game device");
    fixture.load_native(directory);
    fixture.setup(sdk);

    struct foreground_scope {
      lifecycle_fixture &fixture;

      ~foreground_scope() {
        fixture.set_foreground(nullptr);
      }
    } foreground {fixture};

    // The first game device: the FX boot and Streamline discovery, until
    // Game 3D renders (no depth yet).
    fixture.require_renders("first game device");
    {
      auto work = fixture.hold_pool_work();
      require(fixture.end_device("first game device"), "The first game device outlived the game's release");
      fixture.after_unload(*work, "first game device");
    }
    process_usage::now().print("after-first-game-device");
    // Another swapchain size: its shader entries compile on the pool.
    width = 960;
    height = 540;
    brief_device(fixture, "compiling device", [&] {
      return fixture.shader_entries_pending(nullptr) != 0;
    });
    width = 1280;
    height = 720;
    // The game's size, compiled: its renderer builds on the pool, each step
    // delayed so that the device ends mid-build.
    {
      fixture.renderer_step_delay(40);
      unsigned started {}, in_flight {};
      fixture.renderer_builds(&started, nullptr);
      const auto before = started;
      brief_device(fixture, "building device", [&] {
        fixture.renderer_builds(&started, &in_flight);
        return started > before && in_flight != 0;
      });
      fixture.renderer_step_delay(0);
    }
    process_usage::now().print("after-building-device");
    // Game devices verified, then ended with frames in flight. Each outlives
    // the game's release (known gap L1), so the next is the same device: a
    // further cycle must leave the process within a bound of the one before.
    const char *const sessions[] {"second game device", "third game device", "fourth game device"};
    process_usage ended[std::size(sessions)];
    for (std::size_t i = 0; i != std::size(sessions); ++i) {
      fixture.game_device(sessions[i]);
      fixture.verify(sessions[i]);
      fixture.leave_frames_in_flight(sessions[i]);
      auto work = fixture.hold_pool_work();
      work->released = GetTickCount64();
      if (fixture.end_device(sessions[i])) {
        fixture.after_unload(*work, sessions[i]);
      } else {
        // No unload: the held work completes on the pool as ever.
        fixture.release_held(*work, sessions[i]);
      }
      ended[i] = process_usage::now();
      ended[i].print(sessions[i]);
    }
    if (!process_usage::bounded(ended[1], ended[2])) {
      throw std::runtime_error("A game device's further cycle left handles, threads, GDI or USER objects or private memory beyond the bound of the cycle before");
    }
    require(fixture.witnessed == std::size(sessions), "A game device did not witness its known gap");
    std::printf("MEASURE addon-lifecycle gaps %s={witnessed=%u}\n", captured_device_gap.id, fixture.witnessed);
    reshade::unregister_event<reshade::addon_event::reshade_present>(tally_present);
    reshade::unregister_event<reshade::addon_event::destroy_device>(probe_destroy_device);
    reshade::unregister_event<reshade::addon_event::reshade_present>(record_present);
    reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
    std::printf("PASS add-on lifecycle: %u unloads in one process, each leaving the module mapped, its pool work and log lines complete and its device ended, Game 3D rendering on each game device after its reload; %u game devices rendering depth, FG depth and UI after each recreation (known gap %s)\n", fixture.unloads, fixture.witnessed, captured_device_gap.id);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
