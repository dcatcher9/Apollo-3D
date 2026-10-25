// SPDX-License-Identifier: GPL-3.0-only
// ReShade unloads its add-ons whenever a game destroys its last device and
// loads them again for the next one. Assassin's Creed Shadows creates and
// destroys a D3D12 device at startup, while module discovery and a shader
// compile the add-on had started on pool threads were still running its code:
// the game crashed executing the unmapped add-on or D3DCompiler. Here a
// short-lived startup device (no queue, swapchain or frame) starts the
// add-on's real device-initialization pool work and one held pool callback of
// its own code. Its destruction must leave the add-on mapped, the held work
// must then complete and return, and the same module, initialized again by
// ReShade, must render native Game 3D for the game's device.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"

namespace {
  constexpr wchar_t addon_file[] = L"SunshineSBSTest.addon64";

  struct addon_unload_fixture : raw_runtime_fixture {
    using hold_t = BOOL (*)(HANDLE, HANDLE, volatile LONG *);
    native_game3d_observer game3d{*this};
    HMODULE startup_module = nullptr;
    volatile LONG held_compiled = 0;

    // Lines of the isolated runtime's ReShade.log containing text. ReShade
    // writes its own lines synchronously, through to disk.
    std::size_t log_count(const char *text) const {
      std::size_t count{};
      std::ifstream file(runtime_directory / "ReShade.log", std::ios::binary);
      for (std::string line; std::getline(file, line);)
        if (line.find(text) != std::string::npos) ++count;
      return count;
    }

    void startup_device() {
      const auto create = reinterpret_cast<decltype(&D3D12CreateDevice)>(GetProcAddress(runtime_module, "D3D12CreateDevice"));
      require(create, "Official runtime lacks the D3D12 proxy export");
      require(!GetModuleHandleW(addon_file), "The add-on was loaded before any device existed");
      const HANDLE proceed = CreateEventW(nullptr, TRUE, FALSE, nullptr), returned = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      require(proceed && returned, "Could not create the held pool work's events");
      std::uint64_t released{};
      {
        com_ptr<ID3D12Device> device;
        checked(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())), "Create the short-lived startup device");
        startup_module = GetModuleHandleW(addon_file);
        require(startup_module && log_count("Registered add-on \"Sunshine SBS TEST ONLY\"") == 1,
          "ReShade did not load and initialize the add-on for the startup device");
        // The device's initialization already started the add-on's module
        // discovery and content-sampler compile on pool threads; this one is
        // held until after the unload, so it is certainly still to run.
        const auto hold = reinterpret_cast<hold_t>(GetProcAddress(startup_module, "SunshineSbsTestHoldPoolWork"));
        require(hold && hold(proceed, returned, &held_compiled), "Could not start held pool work in the add-on");
        released = GetTickCount64();
      }
      require(log_count("Unloading add-on \"Sunshine SBS TEST ONLY\"") == 1 &&
        log_count("Unregistered add-on \"Sunshine SBS TEST ONLY\"") == 1,
        "ReShade did not uninitialize and unload the add-on with the startup device");
      require(GetModuleHandleW(addon_file) == startup_module,
        "ReShade's unload unmapped the add-on while its pool work was still to run; AddonInit must pin the module");
      SetEvent(proceed);
      require(WaitForSingleObject(returned, 30000) == WAIT_OBJECT_0, "The add-on's held pool work did not return after the unload");
      require(InterlockedCompareExchange(&held_compiled, 0, 0) == 1, "The add-on's held pool compile failed after the unload");
      std::printf("MEASURE addon-unload startup_device_unload_to_pool_return_ms=%llu\n",
        static_cast<unsigned long long>(GetTickCount64() - released));
      CloseHandle(proceed);
      CloseHandle(returned);
      std::puts("PASS the add-on stays mapped across ReShade's unload with its startup device, and its held pool compile completes");
    }

    void run() {
      require(startup_module, "The startup device phase did not run");
      create_pipeline();
      scene = target(width, height, 1, 1);
      render_tracked_depth = [&] { draw(*scene); };
      const auto initialized = GetTickCount64() + 45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64() < initialized) step();
      require(observed.runtime && observed.renders && !observed.inject, "The official runtime did not initialize the game's device");
      check_unified_addon();
      require(GetModuleHandleW(addon_file) == startup_module && log_count("Registered add-on \"Sunshine SBS TEST ONLY\"") == 2,
        "ReShade did not initialize the same pinned add-on module again for the game's device");
      game3d.start();
      game3d.set_strength(100);
      auto sequence = game3d.await_render([&] { step(); });
      // Native Game 3D of the reinitialized add-on: every Present renders, and
      // the drawn depth becomes current, ready and calibrated.
      const auto source = reinterpret_cast<std::uint64_t>(scene->texture.p);
      const auto until = GetTickCount64() + 15000;
      unsigned consecutive = 0;
      sunshine_game3d::test::last_render frame;
      automatic_status status;
      do {
        step(); game3d.no_effects();
        frame = game3d.last_render();
        require(frame.sequence > sequence, "A Present of the game's device had no native Game 3D render");
        sequence = frame.sequence;
        status = game3d.automatic();
        const bool working = frame.rendered && frame.depth.ready && frame.depth.source_resource.handle == source &&
          frame.parameters.depth_ready && status.ready() && status.active_scale();
        consecutive = working ? consecutive + 1 : 0;
      } while (consecutive < 16 && GetTickCount64() < until);
      std::printf("MEASURE addon-unload reinitialized flags=%u scale_state=%u H=%.9g depth_ready=%d rendered=%d\n",
        status.flags, status.scale_state, status.scale, frame.depth.ready ? 1 : 0, frame.rendered ? 1 : 0);
      require(consecutive >= 16, "The reinitialized add-on did not render native Game 3D from the game's drawn depth");
      render_tracked_depth = {};
      std::puts("PASS the reinitialized add-on renders native Game 3D for the game's device from its drawn depth");
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5) {
    std::fputs("usage: reshade_addon_unload_runtime_test <ReShade64.dll> <frozenShaders> <SunshineSBSTest.addon64> <fresh-output-directory>\n", stderr);
    return 2;
  }
  std::thread([] {
    Sleep(120000);
    std::fputs("FAIL add-on unload runtime watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    select_native_boot();
    width = 1280;
    height = 720;
    require(!fs::exists(fs::absolute(argv[4])), "The add-on unload fixture requires a fresh isolated output directory");
    addon_unload_fixture fixture;
    fixture.runtime_directory = fs::absolute(argv[4]);
    fixture.before_game_device = [&] { fixture.startup_device(); };
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fixture.runtime_directory, 2, 0, fs::absolute(argv[3]));
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
