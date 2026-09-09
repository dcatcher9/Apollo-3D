/**
 * @file tests/unit/test_process.cpp
 * @brief Unit tests for application command handling.
 */
#include "../tests_common.h"

// standard includes
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

// lib includes
#include <boost/process/v1/environment.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/token_functions.hpp>

// local includes
#include "src/config.h"
#include "src/process.h"

namespace proc {
  // A retained desktop fixture with no real process, display, or HDR worker. The successful
  // same-mode apply returns before any Windows call, exercising transport admission itself.
  struct process_test_access {
    static void retain(proc_t &process, const std::shared_ptr<rtsp_stream::launch_session_t> &launch) {
      process._app_id = 1;
      process._host_session_id = 1234;
      process._launch_session = launch;
      process._active_launch_session_id = launch->id;
#ifdef _WIN32
      process._virtual_display_only = launch->virtual_display_only;
#endif
    }
#ifdef _WIN32
    static void mark_virtual(proc_t &process, bool enabled, bool with_identity = false) {
      process._virtual_display = enabled;
      process._virtual_display_gdi_name = enabled ? L"test-only-display" : L"";
      process._virtual_display_device_path = enabled ? L"test-only-monitor-path" : L"";
      if (enabled && with_identity) {
        process._virtual_display_identity.emplace();
      } else {
        process._virtual_display_identity.reset();
      }
    }

    static bool virtual_display_only(const proc_t &process) {
      return process._virtual_display_only;
    }

    static void set_display_topology_hook(
      proc_t &process,
      display_topology_test_hook_t hook
    ) {
      process._display_topology_test_hook = std::move(hook);
    }
#endif
    static void clear(proc_t &process) {
      process._app_id = 0;
      process._host_session_id = 0;
      process._active_launch_session_id = 0;
      process._launch_session.reset();
#ifdef _WIN32
      process._display_topology_test_hook = {};
      mark_virtual(process, false);
      process._virtual_display_only = false;
#endif
    }
  };
}

TEST(ProcessTest, ResumePublishesNewLiveTransportWithoutChangingRetainedToken) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  proc::process_test_access::retain(process, original);
  auto cleanup = util::fail_guard([&]() { proc::process_test_access::clear(process); });

  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  resumed->id = 22;
  resumed->width = 1920;
  resumed->height = 1080;
  resumed->fps = 60000;
  resumed->scale_factor = 100;
  EXPECT_EQ(process.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  EXPECT_EQ(original->id, 11U);
#ifdef _WIN32
  proc::process_test_access::mark_virtual(process, true);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 22), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 11), proc::live_video_mode_result_e::needs_reconnect);
  EXPECT_TRUE(process.live_video_mode_needs_display_change(1920, 1080, 90000));
  EXPECT_TRUE(process.live_video_mode_needs_display_change(1280, 720, 60000));
  EXPECT_FALSE(process.live_video_mode_needs_display_change(1920, 1080, 60000));
  proc::process_test_access::mark_virtual(process, false);
#endif
  auto invalid = std::make_shared<rtsp_stream::launch_session_t>();
  invalid->id = 33;
  invalid->width = 1;
  invalid->height = 1;
  invalid->scale_factor = 20;
  EXPECT_EQ(process.reconfigure_retained_session(invalid), 400);
#ifdef _WIN32
  proc::process_test_access::mark_virtual(process, true);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 22), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 33), proc::live_video_mode_result_e::needs_reconnect);
#endif
}

#ifdef _WIN32
TEST(ProcessTest, RetainedDisplayPolicyFollowsEachAcceptedClientResume) {
  const auto saved_output = config::video.output_name;
  auto restore_config = util::fail_guard([&]() {
    config::video.output_name = saved_output;
  });

  for (const bool captured_policy : {false, true}) {
    SCOPED_TRACE(captured_policy);
    proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
    auto original = std::make_shared<rtsp_stream::launch_session_t>();
    original->id = 11;
    original->width = 1920;
    original->height = 1080;
    original->fps = 60000;
    original->virtual_display_only = captured_policy;
    proc::process_test_access::retain(process, original);
    auto cleanup = util::fail_guard([&]() {
      proc::process_test_access::clear(process);
    });

    // This fixture owns no display, so successful retained reconfiguration and teardown exercise
    // client policy lifetime without any CCD call.
    auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
    resumed->id = 22;
    resumed->width = 1920;
    resumed->height = 1080;
    resumed->fps = 60000;
    resumed->scale_factor = 100;
    resumed->virtual_display_only = !captured_policy;
    ASSERT_EQ(process.reconfigure_retained_session(resumed), 0);
    EXPECT_EQ(proc::process_test_access::virtual_display_only(process), !captured_policy);

    auto rejected = std::make_shared<rtsp_stream::launch_session_t>();
    rejected->id = 23;
    rejected->width = 1920;
    rejected->height = 1080;
    rejected->fps = 60000;
    rejected->scale_factor = 100;
    rejected->virtual_display = true;
    rejected->virtual_display_only = captured_policy;
    EXPECT_EQ(process.reconfigure_retained_session(rejected), 409);
    EXPECT_EQ(proc::process_test_access::virtual_display_only(process), !captured_policy);

    auto invalid = std::make_shared<rtsp_stream::launch_session_t>();
    invalid->width = 1;
    invalid->height = 1;
    invalid->scale_factor = 20;
    invalid->virtual_display_only = captured_policy;
    EXPECT_EQ(process.reconfigure_retained_session(invalid), 400);
    EXPECT_EQ(proc::process_test_access::virtual_display_only(process), !captured_policy);

    process.terminate(false, false);
    EXPECT_FALSE(proc::process_test_access::virtual_display_only(process));
    EXPECT_EQ(process.get_host_session_id(), 0U);
  }
}

TEST(ProcessTest, RetainedVirtualDisplayResumeCommitsPolicyAfterPromotionAndRebind) {
  using operation_e = proc::display_topology_test_operation_e;

  for (const bool previous_policy : {false, true}) {
    SCOPED_TRACE(previous_policy);
    proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
    auto original = std::make_shared<rtsp_stream::launch_session_t>();
    original->id = 11;
    original->width = 1920;
    original->height = 1080;
    original->fps = 60000;
    original->virtual_display = true;
    original->virtual_display_only = previous_policy;
    proc::process_test_access::retain(process, original);
    proc::process_test_access::mark_virtual(process, true);
    auto cleanup = util::fail_guard([&]() {
      proc::process_test_access::clear(process);
    });

    std::vector<std::pair<operation_e, bool>> operations;
    proc::process_test_access::set_display_topology_hook(
      process,
      [&](operation_e operation, bool value) {
        operations.emplace_back(operation, value);
        return true;
      }
    );

    auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
    resumed->id = 22;
    resumed->width = 1920;
    resumed->height = 1080;
    resumed->fps = 60000;
    resumed->scale_factor = 100;
    resumed->virtual_display_only = !previous_policy;
    ASSERT_EQ(process.reconfigure_retained_session(resumed), 0);

    ASSERT_EQ(operations.size(), 4U);
    EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, !previous_policy}));
    EXPECT_EQ(operations[1], (std::pair {operation_e::request_hdr, false}));
    EXPECT_EQ(operations[2], (std::pair {operation_e::promote, !previous_policy}));
    EXPECT_EQ(operations[3], (std::pair {operation_e::refresh_binding, !previous_policy}));
    EXPECT_EQ(proc::process_test_access::virtual_display_only(process), !previous_policy);
    EXPECT_EQ(original->virtual_display_only, !previous_policy);
    EXPECT_TRUE(resumed->virtual_display);
    EXPECT_EQ(process.get_host_session_id(), 1234U);
  }
}

TEST(ProcessTest, FailedPostPromotionRebindRequestsRetirementAndClearsSession) {
  using operation_e = proc::display_topology_test_operation_e;

  const auto saved_output = config::video.output_name;
  auto restore_config = util::fail_guard([&]() {
    config::video.output_name = saved_output;
  });

  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  process.initial_display = saved_output;
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  original->virtual_display = true;
  original->virtual_display_only = false;
  proc::process_test_access::retain(process, original);
  proc::process_test_access::mark_virtual(process, true, true);
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });

  std::vector<std::pair<operation_e, bool>> operations;
  int refresh_count = 0;
  proc::process_test_access::set_display_topology_hook(
    process,
    [&](operation_e operation, bool value) {
      operations.emplace_back(operation, value);
      if (operation == operation_e::refresh_binding) {
        // The retained binding resolves before promotion, but its required post-promotion
        // verification fails after Windows accepted the topology change.
        return ++refresh_count == 1;
      }
      return true;
    }
  );

  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  resumed->id = 22;
  resumed->width = 1920;
  resumed->height = 1080;
  resumed->fps = 60000;
  resumed->scale_factor = 100;
  resumed->virtual_display_only = true;
  EXPECT_EQ(process.reconfigure_retained_session(resumed), 503);

  ASSERT_EQ(operations.size(), 5U);
  EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[1], (std::pair {operation_e::request_hdr, false}));
  EXPECT_EQ(operations[2], (std::pair {operation_e::promote, true}));
  EXPECT_EQ(operations[3], (std::pair {operation_e::refresh_binding, true}));
  // The retirement implementation has lower-level coverage; this process test verifies that a
  // failed post-promotion rebind requests final retirement before clearing the retained session.
  EXPECT_EQ(operations[4], (std::pair {operation_e::retire, true}));
  EXPECT_FALSE(proc::process_test_access::virtual_display_only(process));
  EXPECT_FALSE(original->virtual_display_only);
  EXPECT_EQ(process.get_host_session_id(), 0U);
  const auto status = process.get_status();
  EXPECT_EQ(status.app_id, 0);
  EXPECT_FALSE(status.virtual_display);
}

TEST(ProcessTest, RejectedNewLaunchClearsPreviousClientDisplayPolicy) {
  const auto saved_output = config::video.output_name;
  auto restore_config = util::fail_guard([&]() {
    config::video.output_name = saved_output;
  });

  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto previous = std::make_shared<rtsp_stream::launch_session_t>();
  previous->id = 11;
  previous->virtual_display_only = true;
  proc::process_test_access::retain(process, previous);
  auto cleanup = util::fail_guard([&]() { proc::process_test_access::clear(process); });

  proc::ctx_t app {};
  app.id = "1";
  app.name = "Invalid exclusive mode test";
  auto invalid = std::make_shared<rtsp_stream::launch_session_t>();
  invalid->width = 1;
  invalid->height = 1;
  invalid->scale_factor = 20;
  invalid->virtual_display_only = true;
  EXPECT_EQ(process.execute(app, invalid, false), 400);
  EXPECT_FALSE(proc::process_test_access::virtual_display_only(process));
  EXPECT_EQ(process.get_host_session_id(), 0U);
}
#endif

TEST(ProcessTest, ExplorerRepairIsOptInByDefault) {
  EXPECT_FALSE(config::default_virtual_display_restart_explorer);
}

TEST(ProcessTest, CalculatesEvenScaledRenderDimensions) {
  EXPECT_EQ(proc::calculate_render_size(5120, 2160, 100), (proc::render_size_t {5120, 2160}));
  EXPECT_EQ(proc::calculate_render_size(3552, 3840, 125), (proc::render_size_t {4440, 4800}));
  EXPECT_EQ(proc::calculate_render_size(3552, 3840, 75), (proc::render_size_t {2664, 2880}));
  EXPECT_EQ(proc::calculate_render_size(1921, 1081, 100), (proc::render_size_t {1920, 1080}));
  EXPECT_EQ(proc::calculate_render_size(1920, 1080, 20), (proc::render_size_t {384, 216}));
  EXPECT_FALSE(proc::calculate_render_size(1, 1, 100));
  EXPECT_FALSE(proc::calculate_render_size(1920, 1080, 0));
}

TEST(ProcessTest, InvalidRenderSizeLeavesProcessIdle) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  proc::ctx_t app {};
  app.id = "1";
  app.name = "Invalid mode test";

  auto launch = std::make_shared<rtsp_stream::launch_session_t>();
  launch->width = 1;
  launch->height = 1;
  launch->fps = 60000;
  launch->scale_factor = 20;

  EXPECT_EQ(process.execute(app, launch, false), 400);
  const auto status = process.get_status();
  EXPECT_EQ(status.app_id, 0);
  EXPECT_EQ(status.host_session_id, 0U);
}

TEST(ProcessTest, LiveVideoModeIsRefusedWithoutAVirtualDisplay) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};

  // An idle host owns no desktop it may resize, so the change can only come from a fresh launch.
  EXPECT_EQ(
    process.apply_live_video_mode(1920, 1080, 60000, 1),
    proc::live_video_mode_result_e::needs_reconnect
  );

  // Nonsense geometry never reaches the Windows topology.
  EXPECT_EQ(
    process.apply_live_video_mode(0, 1080, 60000, 1),
    proc::live_video_mode_result_e::needs_reconnect
  );
  EXPECT_EQ(
    process.apply_live_video_mode(1920, 1080, 0, 1),
    proc::live_video_mode_result_e::needs_reconnect
  );

  // The control thread's fast-path hint must say "no display work" so a bitrate-only change is
  // never queued behind a topology transition that would not happen anyway.
  EXPECT_FALSE(process.live_video_mode_needs_display_change(1920, 1080, 60000));
}

TEST(ProcessTest, InputConfinementIdentityRequiresOwnedRemoteVirtualDisplay) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  EXPECT_TRUE(process.virtual_display_device_path().empty());
#ifdef _WIN32
  auto launch = std::make_shared<rtsp_stream::launch_session_t>();
  launch->id = 47;
  proc::process_test_access::retain(process, launch);
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });
  proc::process_test_access::mark_virtual(process, true);
  EXPECT_TRUE(process.virtual_display_device_path().empty());
  launch->virtual_display = true;
  EXPECT_EQ(process.virtual_display_device_path(), "test-only-monitor-path");
  proc::process_test_access::mark_virtual(process, false);
  EXPECT_TRUE(process.virtual_display_device_path().empty());
#endif
}

TEST(ProcessTest, LiveVideoModeFailureIsRetryableOnlyAfterProvenRollback) {
  EXPECT_EQ(
    proc::live_video_mode_failure_result(true),
    proc::live_video_mode_result_e::failed
  );
  EXPECT_EQ(
    proc::live_video_mode_failure_result(false),
    proc::live_video_mode_result_e::needs_reconnect
  );
}

#ifdef _WIN32
TEST(ProcessTest, DriverRemovalRequiresConfirmedDetachWhenDesktopDeactivationWasRequested) {
  EXPECT_FALSE(proc::retiredVirtualDisplayRemovalAllowedForTest(true, false));
  EXPECT_TRUE(proc::retiredVirtualDisplayRemovalAllowedForTest(true, true));
  EXPECT_TRUE(proc::retiredVirtualDisplayRemovalAllowedForTest(false, false));
  EXPECT_TRUE(proc::retiredVirtualDisplayRemovalAllowedForTest(false, true));
}

TEST(ProcessTest, RetirementHandoffMarksOnlyTheBoundDisplayOrAnUnboundCandidate) {
  EXPECT_TRUE(proc::virtualDisplayRetirementHandoffMarksSessionForTest(false, false));
  EXPECT_TRUE(proc::virtualDisplayRetirementHandoffMarksSessionForTest(false, true));
  EXPECT_TRUE(proc::virtualDisplayRetirementHandoffMarksSessionForTest(true, true));
  EXPECT_FALSE(proc::virtualDisplayRetirementHandoffMarksSessionForTest(true, false));
}

TEST(ProcessTest, ExplorerRepairRequiresEveryFinalRetirementProof) {
  EXPECT_TRUE(proc::explorerRepairAllowedForRetirementForTest(
    true,
    true,
    true
  ));

  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(
    false,
    true,
    true
  )) << "Warm retirement must retain debt without restarting Explorer";
  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(
    true,
    false,
    true
  )) << "The opt-in config must remain suppressible";
  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(
    true,
    true,
    false
  )) << "Cleanup of a path that was never active creates no repair debt";
}
#endif

TEST(ProcessTest, MalformedCommandDoesNotEscapeWorkingDirectoryResolution) {
  const std::string malformed_command {"command\\"};
  const auto env = boost::this_process::environment();

#ifdef _WIN32
  // split_winmain intentionally accepts a trailing backslash using Windows command-line rules.
  EXPECT_NO_THROW((void) boost::program_options::split_winmain(malformed_command));
  EXPECT_NO_THROW((void) proc::find_working_directory(malformed_command, env));
#else
  EXPECT_THROW(boost::program_options::split_unix(malformed_command), boost::escaped_list_error);
  EXPECT_NO_THROW({
    const auto working_directory = proc::find_working_directory(malformed_command, env);
    EXPECT_TRUE(working_directory.empty());
  });
#endif
}

#ifdef _WIN32
TEST(ProcessTest, PlatformLaunchDoesNotReuseStaleErrorCode) {
  const auto env = boost::this_process::environment();
  boost::filesystem::path working_directory;
  const auto stale_error = std::make_error_code(std::errc::permission_denied);
  std::error_code ec = stale_error;

  auto child = platf::run_command(
    false,
    false,
    "apollo_command_that_must_not_exist_7f42e31b",
    working_directory,
    env,
    nullptr,
    ec,
    nullptr
  );

  EXPECT_FALSE(child.valid());
  EXPECT_TRUE(ec);
  EXPECT_NE(ec, stale_error);
  EXPECT_EQ(ec.category(), std::system_category());
  EXPECT_EQ(ec.value(), ERROR_FILE_NOT_FOUND);
}

TEST(ProcessTest, AddsCanonicalVirtualDisplayTileWhenDriverIsReady) {
  const auto previous_status = proc::vDisplayDriverStatus.exchange(VDISPLAY::DRIVER_STATUS::OK);
  const auto restore_status = util::fail_guard([previous_status]() {
    proc::vDisplayDriverStatus.store(previous_status);
  });

  const auto apps_path = std::filesystem::temp_directory_path() /
                         "apollo_virtual_display_test_apps.json";
  const auto remove_apps = util::fail_guard([&apps_path]() {
    std::error_code ec;
    std::filesystem::remove(apps_path, ec);
  });
  {
    std::ofstream apps_file(apps_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(apps_file.is_open());
    apps_file << R"({
      "version": 2,
      "env": {},
      "apps": [{
        "uuid": "324E9C05-F190-4439-B9C4-20B1C8C12DF6",
        "name": "Desktop",
        "image-path": "desktop.png"
      }]
    })";
  }

  auto parsed = proc::parse(apps_path.string());
  ASSERT_TRUE(parsed.has_value());
  const auto apps = parsed->get_apps();
  ASSERT_EQ(apps.size(), 2u);

  const auto virtual_app = std::find_if(apps.begin(), apps.end(), [](const proc::ctx_t &app) {
    return app.uuid == VIRTUAL_DISPLAY_UUID;
  });
  ASSERT_NE(virtual_app, apps.end());
  EXPECT_EQ(virtual_app->name, "Virtual Display");
  EXPECT_EQ(virtual_app->image_path, "virtual_desktop.png");
  EXPECT_TRUE(virtual_app->synthetic_virtual_display);
  EXPECT_TRUE(virtual_app->cmd.empty());

  const auto desktop_app = std::find_if(apps.begin(), apps.end(), [](const proc::ctx_t &app) {
    return app.name == "Desktop";
  });
  ASSERT_NE(desktop_app, apps.end());
  EXPECT_FALSE(desktop_app->synthetic_virtual_display);
  EXPECT_NE(desktop_app->id, virtual_app->id);
}
#endif
