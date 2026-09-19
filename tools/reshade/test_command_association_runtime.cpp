// SPDX-License-Identifier: GPL-3.0-only
// Actual ReShade lifecycle and preserved-depth-copy wiring. Evaluation markers
// here are synthetic; the separate MinHook fixture exercises Streamline calls.
#define SUNSHINE_COMMAND_ASSOCIATION_RUNTIME
#include "test_depth_selection_runtime.cpp"
#include "depth_addon.h"
#include "native_discard_observer.h"

namespace {
  namespace command_evidence = sunshine_streamline::commands;
  using capture_fn = BOOL (*)(std::uint64_t, command_evidence::recording_marker *);
  using associate_fn = BOOL (*)(const command_evidence::recording_marker *, const command_evidence::recording_marker *,
    std::uint64_t, command_evidence::association *);
  using frame_fn = BOOL (*)(api::effect_runtime *, sunshine_depth::frame_depth *);
  capture_fn capture_marker{};
  associate_fn associate_markers{};
  frame_fn read_frame{};
  using use_fn = BOOL (*)(std::uint64_t, std::uint64_t, std::uint64_t, sunshine_streamline::content::use_snapshot *);
  using content_fn = BOOL (*)(const sunshine_streamline::content::use_snapshot *, const sunshine_streamline::content::copy_snapshot *,
    std::uint64_t, std::uint64_t, sunshine_streamline::content::association *);
  use_fn capture_content_use{};
  content_fn associate_content{};
  using discard_counts_fn = BOOL (*)(sunshine_streamline::native_discard::counters *);
  discard_counts_fn discard_counts{};
  sunshine_streamline::content::use_snapshot content_use{};
  sunshine_streamline::content::association content_association{};
  void (*lose_lifecycle)(){};
  std::uint64_t reset_native{};
  command_evidence::recording_marker evaluation_marker{};
  command_evidence::association frame_association{};
  sunshine_depth::frame_depth captured_frame{};
  bool frame_ready{};

  void observe_reset(api::command_list *commands) {
    reset_native = commands->get_native();
  }
  void observe_depth_frame(api::effect_runtime *runtime, api::command_list *, api::resource_view, api::resource_view) {
    frame_ready = false;
    captured_frame = {};
    frame_association = {};
    content_association = {};
    if (!read_frame || !associate_markers || !read_frame(runtime, &captured_frame)) return;
    frame_ready = associate_markers(&evaluation_marker, &captured_frame.capture_marker,
      captured_frame.command_queue, &frame_association) != FALSE;
    if (associate_content) associate_content(&content_use, &captured_frame.depth_copy,
      captured_frame.resource.handle, captured_frame.backup_id, &content_association);
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
      read_frame = reinterpret_cast<frame_fn>(GetProcAddress(module, "SunshineDepthTestFrame"));
      capture_content_use = reinterpret_cast<use_fn>(GetProcAddress(module, "SunshineContentTestUse"));
      associate_content = reinterpret_cast<content_fn>(GetProcAddress(module, "SunshineContentTestAssociate"));
      const auto enable_content = reinterpret_cast<void (*)(BOOL)>(GetProcAddress(module, "SunshineContentTestEnable"));
      const auto install_discard = reinterpret_cast<BOOL (*)()>(GetProcAddress(module, "SunshineDiscardTestInstallPending"));
      discard_counts = reinterpret_cast<discard_counts_fn>(GetProcAddress(module, "SunshineDiscardTestCounts"));
      lose_lifecycle = reinterpret_cast<void (*)()>(GetProcAddress(module, "SunshineCommandTestLoseLifecycle"));
      require(capture_marker && associate_markers && read_frame && lose_lifecycle && capture_content_use && associate_content && enable_content && install_discard && discard_counts,
        "Test-only observation adapters are missing");
      enable_content(TRUE);
      require(install_discard(), "Deferred native DiscardResource hook installation failed");
      sunshine_streamline::native_discard::counters installed;
      require(discard_counts(&installed) && installed.installed > 0, "Actual native DiscardResource target was not observed at command initialization");
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
      reshade::register_event<reshade::addon_event::reshade_begin_effects>(observe_depth_frame);
    }
    void run() {
      load_adapter();
      create_pipeline();
      scene = target(width, height, 1, 1, false, true);
      bool evaluate_after_copy = false;
      render_tracked_depth = [&] {
        if (!evaluate_after_copy) evaluation_marker = capture(reset_native);
        draw(*scene);
        if (evaluate_after_copy) evaluation_marker = capture(reset_native);
      };
      const auto load_limit = std::chrono::steady_clock::now() + std::chrono::seconds(35);
      while ((!observed.runtime || !observed.renders) && std::chrono::steady_clock::now() < load_limit) step();
      require(observed.runtime && observed.renders, "Original-derived effect did not compile");
      require(!observed.inject, "Association fixture must use actual selected DSV copies");
      set_int("Depth_Map", 1);
      set_float("Depth_Map_Adjust", 40);
      for (const bool after : {false, true}) {
        evaluate_after_copy = after;
        unsigned accepted = 0;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(12);
        while (accepted < 4 && std::chrono::steady_clock::now() < until) {
          step();
          if (!frame_ready || !frame_association.associated()) continue;
          require(captured_frame.source_resource.handle == reinterpret_cast<std::uint64_t>(scene->texture.p),
            "Recording marker belongs to a different selected depth source");
          require(captured_frame.capture_marker.command == evaluation_marker.command && frame_association.same_recording,
            "Actual pre-clear copy was not attached to its source command recording");
          require(frame_association.ordering == (after ? command_evidence::order::copy_before_evaluation :
            command_evidence::order::evaluation_before_copy), "Preserved-copy event ordering was reversed");
          require(!frame_association.content_registered && !frame_association.final_color_registered,
            "Actual-runtime ordering was incorrectly promoted into content correspondence");
          ++accepted;
        }
        if (accepted < 4) std::fprintf(stderr, "Association status: %s ready=%d marker=%llu eval=%llu\n",
          command_evidence::name(frame_association.state), int(frame_ready),
          static_cast<unsigned long long>(captured_frame.capture_marker.command),
          static_cast<unsigned long long>(evaluation_marker.command));
        require(accepted == 4, "Actual selected-depth command association did not become available");
        std::printf("PASS actual pre-clear depth copy: %s; source and recording match; content/color remain unverified\n",
          after ? "copy-before-evaluation" : "evaluation-before-copy");
      }
      require(varied_linear_depth(), "Recording instrumentation changed the selected shader-visible depth");
      const auto source_id = captured_frame.source_id;
      const auto capture_use = [&] {
        evaluation_marker = capture(reset_native);
        require(capture_content_use(reset_native, reinterpret_cast<std::uint64_t>(scene->texture.p), source_id, &content_use),
          "Content-use adapter rejected valid storage");
      };
      const char *labels[] {"evaluation-copy-clear", "copy-clear-evaluation", "evaluation-draw-copy", "copy-evaluation-unchanged",
        "copy-native-discard-evaluation", "copy-native-partial-discard-evaluation"};
      for (unsigned scenario = 0; scenario != 6; ++scenario) {
        scene->clear_after = scenario < 3;
        scene->before_final_clear = scenario == 0 ? std::function<void()>(capture_use) : std::function<void()>();
        render_tracked_depth = [&] {
          if (scenario == 2) capture_use();
          draw(*scene);
          if (scenario >= 4) {
            // Valid native D3D12 discard of a DEPTH_WRITE resource, after the
            // add-on has preserved its contents. ReShade 6.8 emits no callback.
            // The optional region is passed unchanged through both wrappers.
            const D3D12_RECT rect {0, 0, LONG(width / 2), LONG(height)};
            const D3D12_DISCARD_REGION region {1, &rect, 0, 1};
            sunshine_streamline::native_discard::counters before, after;
            require(discard_counts(&before), "Cannot read native discard counters");
            commands->DiscardResource(scene->texture.p, scenario == 5 ? &region : nullptr);
            require(discard_counts(&after) && after.observed > before.observed,
              "Actual native DiscardResource bypassed the depth-content observer");
          }
          if (scenario == 1 || scenario >= 3) capture_use();
        };
        unsigned checked_frames = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (checked_frames != 4 && std::chrono::steady_clock::now() < deadline) {
          step();
          if (!frame_ready || !frame_association.associated() || (scenario < 4 && !content_use.valid()) || !captured_frame.depth_copy.valid()) continue;
          require(!content_association.coverage_complete && !content_association.final_color_registered,
            "Observed depth mutations were promoted into complete native coverage");
          const bool expected_match = scenario == 0 || scenario == 3;
          if (content_association.matched() != expected_match)
            std::fprintf(stderr, "Content case %s: %s\n", labels[scenario], sunshine_streamline::content::name(content_association.state));
          require(content_association.matched() == expected_match, "Actual source content/copy association disagrees with draw/clear ordering");
          ++checked_frames;
        }
        if (checked_frames != 4) std::fprintf(stderr, "Content timeout %s: use=%s copy=%s association=%s\n", labels[scenario],
          sunshine_streamline::content::name(content_use.state), sunshine_streamline::content::name(captured_frame.depth_copy.state),
          sunshine_streamline::content::name(content_association.state));
        require(checked_frames == 4, "Content observations failed to become available");
        std::printf("PASS actual DSV content ordering: %s; native coverage remains explicitly incomplete\n", labels[scenario]);
      }
      scene->before_final_clear = {};
      render_tracked_depth = {};

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
      reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(observe_depth_frame);
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
