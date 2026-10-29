// SPDX-License-Identifier: GPL-3.0-only
// Actual ReShade command-list lifecycle wiring and the native DiscardResource
// hook installation. Evaluation markers here are synthetic; the separate
// MinHook fixture exercises Streamline calls.
#define SUNSHINE_COMMAND_ASSOCIATION_RUNTIME
#include "test_depth_selection_runtime.cpp"
#include "depth_addon.h"
#include "native_discard_observer.h"

namespace {
  namespace command_evidence = sunshine_streamline::commands;
  using capture_fn = BOOL (*)(std::uint64_t, command_evidence::recording_marker *);
  using associate_fn = BOOL (*)(const command_evidence::recording_marker *, const command_evidence::recording_marker *,
    std::uint64_t, command_evidence::association *);
  capture_fn capture_marker{};
  associate_fn associate_markers{};
  using discard_counts_fn = BOOL (*)(sunshine_streamline::native_discard::counters *);
  discard_counts_fn discard_counts{};
  void (*lose_lifecycle)(){};
  std::uint64_t reset_native{};

  void observe_reset(api::command_list *commands) {
    reset_native = commands->get_native();
  }
  command_evidence::recording_marker capture(std::uint64_t native) {
    command_evidence::recording_marker out;
    require(capture_marker(native, &out) && bool(out), "Actual open command recording was not observed");
    return out;
  }
  command_evidence::association associate(const command_evidence::recording_marker &evaluation,
      const command_evidence::recording_marker &copy, std::uint64_t queue) {
    command_evidence::association out;
    require(associate_markers(&evaluation, &copy, queue, &out), "Test query rejected valid output storage");
    require(!out.content_registered && !out.final_color_registered, "Ordering observations were promoted to unverified image correspondence");
    return out;
  }

  struct association_fixture : selection_fixture_t {
    void load_adapter() {
      const auto module = GetModuleHandleW(L"SunshineSBS.addon64");
      require(module != nullptr, "Actual runtime did not load the supplied test add-on");
      const auto label = reinterpret_cast<const char *const *>(GetProcAddress(module, "NAME"));
      require(label && *label && std::strcmp(*label, "Sunshine SBS TEST ONLY") == 0,
        "Association fixture requires the distinct test add-on, never production");
      capture_marker = reinterpret_cast<capture_fn>(GetProcAddress(module, "SunshineCommandTestCapture"));
      associate_markers = reinterpret_cast<associate_fn>(GetProcAddress(module, "SunshineCommandTestAssociate"));
      const auto enable_content = reinterpret_cast<void (*)(BOOL)>(GetProcAddress(module, "SunshineContentTestEnable"));
      const auto install_discard = reinterpret_cast<BOOL (*)()>(GetProcAddress(module, "SunshineDiscardTestInstallPending"));
      discard_counts = reinterpret_cast<discard_counts_fn>(GetProcAddress(module, "SunshineDiscardTestCounts"));
      lose_lifecycle = reinterpret_cast<void (*)()>(GetProcAddress(module, "SunshineCommandTestLoseLifecycle"));
      require(capture_marker && associate_markers && lose_lifecycle && enable_content && install_discard && discard_counts,
        "Test-only observation adapters are missing");
      enable_content(TRUE);
      require(install_discard(), "Deferred native DiscardResource hook installation failed");
      sunshine_streamline::native_discard::counters installed;
      require(discard_counts(&installed) && installed.installed > 0, "Actual native DiscardResource target was not observed at command initialization");
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
    }
    void run() {
      load_adapter();
      const auto load_limit = std::chrono::steady_clock::now() + std::chrono::seconds(35);
      while ((!observed.runtime || !observed.renders) && std::chrono::steady_clock::now() < load_limit) step();
      require(observed.runtime && observed.renders, "Original-derived effect did not compile");

      // Reset and execute actual empty GPU command recordings without presenting.
      // A marker captured while recording becomes eligible only after close/submit.
      begin_commands();
      const auto native = reset_native;
      const auto first = capture(native), second = capture(native);
      const auto native_queue = observed.runtime->get_command_queue()->get_native();
      require(!associate(first, second, native_queue).associated(), "Open recording was accepted");
      submit();
      require(associate(first, second, native_queue).associated(), "Actual close/execute callbacks were not recorded");
      ID3D12CommandList *again[] {commands.p};
      queue->ExecuteCommandLists(1, again);
      checked(queue->Signal(completion.p, ++fence_value), "Fence repeated empty recording");
      checked(completion->SetEventOnCompletion(fence_value, completion_event), "Observe repeated recording completion");
      require(WaitForSingleObject(completion_event, 3000) == WAIT_OBJECT_0, "Repeated recording completion timed out");
      require(associate(first, second, native_queue).state == command_evidence::status::repeated_submission,
        "Repeated actual ExecuteCommandLists was accepted");
      begin_commands();
      const auto fresh = capture(native), fresh_copy = capture(native);
      require(fresh.recording_generation != first.recording_generation && fresh.object_generation == first.object_generation,
        "Actual Reset did not advance the recording generation");
      submit();
      require(associate(fresh, fresh_copy, native_queue).associated(), "New recording failed to recover after reset");
      lose_lifecycle();
      require(!associate(fresh, fresh_copy, native_queue).associated(), "Missed lifecycle retained old evidence");
      begin_commands();
      const auto recovered = capture(native), recovered_copy = capture(native);
      submit();
      require(associate(recovered, recovered_copy, native_queue).associated(),
        "Existing command list/queue did not recover using their private lifetime cookies");
      commands.reset();
      require(!associate(recovered, recovered_copy, native_queue).associated(), "Destroyed command object retained live evidence");
      command_evidence::recording_marker missing;
      require(!capture_marker(0x1234, &missing) && !missing, "Unknown native identity was accepted");
      std::puts("PASS actual ReShade lifecycle, repeated-submit rejection, loss recovery on existing objects, destruction and unknown identity");
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
    }
  };
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 5) {
    std::fputs("usage: command_association_runtime <ReShade64.dll> <Depth3D-Shaders> <SunshineSBSTest.addon64> <isolated-output>\n", stderr);
    return 2;
  }
  std::thread([] { Sleep(100000); std::fputs("FAIL command association runtime watchdog\n", stderr); TerminateProcess(GetCurrentProcess(), 124); }).detach();
  try {
    width = 640; height = 360;
    association_fixture fixture;
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), fs::absolute(argv[4]), 2, 0, fs::absolute(argv[3]));
    fixture.run();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
