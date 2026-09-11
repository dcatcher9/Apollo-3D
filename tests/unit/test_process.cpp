/**
 * @file tests/unit/test_process.cpp
 * @brief Unit tests for application command handling.
 */
#include "../tests_common.h"

// standard includes
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

// lib includes
#include <boost/process/v1/environment.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/token_functions.hpp>
#ifdef _WIN32
  #include <boost/process/v1/io.hpp>
  #include <boost/process/v1/pipe.hpp>
  #include <boost/process/v1/windows.hpp>
#endif

// local includes
#include "src/config.h"
#include "src/process.h"
#include "src/stream.h"

namespace proc {
  // A retained desktop fixture with no real process, display, or HDR worker. The successful
  // same-mode apply returns before any Windows call, exercising transport admission itself.
  struct process_test_access {
    static void retain(proc_t &process, const std::shared_ptr<rtsp_stream::launch_session_t> &launch) {
      process._app = {};
      process.placebo = false;
      process._app_id = 1;
      process._host_session_id = 1234;
      process._launch_session = launch;
      process._active_launch_session_id = launch->id;
#ifdef _WIN32
      process._virtual_display_only = launch->virtual_display_only;
      process._paused_virtual_display.reset();
      process._remote_display_pause_pending = false;
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

    static bool display_pause_pending(const proc_t &process) {
      return process._remote_display_pause_pending;
    }

    static std::wstring virtual_device_path(const proc_t &process) {
      return process._virtual_display_device_path;
    }

    static bool has_virtual_identity(const proc_t &process) {
      return process._virtual_display_identity.has_value();
    }

    static std::uint32_t active_transport(const proc_t &process) {
      return process._active_launch_session_id;
    }

    static void set_display_topology_hook(
      proc_t &process,
      display_topology_test_hook_t hook
    ) {
      process._display_topology_test_hook = std::move(hook);
    }

    static void set_child(proc_t &process, boost::process::v1::child child, bool auto_detach = false) {
      process._process = std::move(child);
      process._app = {};
      process._app.auto_detach = auto_detach;
      process.placebo = false;
      process._app_launch_time = std::chrono::steady_clock::now();
    }

    static void mark_desktop(proc_t &process, bool enabled = true) {
      process.placebo = enabled;
    }

    static void add_undo(proc_t &process, std::string command) {
      process._app.prep_cmds.emplace_back(std::string {}, std::move(command), false);
      process._completed_prep_commands = process._app.prep_cmds.size();
      process._app.exit_timeout = std::chrono::seconds {1};
    }

    static void wait_for_group(proc_t &process, boost::process::v1::group group) {
      process._process_group = std::move(group);
      process._app.wait_all = true;
    }

    static void stop_child(proc_t &process) {
      if (process._process.valid()) {
        std::error_code error;
        if (process._process.running(error)) {
          process._process.terminate(error);
        }
        process._process.wait(error);
      }
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
      process._paused_virtual_display.reset();
      process._remote_display_pause_pending = false;
#endif
    }
  };
}  // namespace proc

TEST(ProcessTest, ResumePublishesNewLiveTransportWithoutChangingRetainedToken) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  proc::process_test_access::retain(process, original);
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });

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
namespace {
  std::string test_command_interpreter() {
    std::array<wchar_t, MAX_PATH> system_directory {};
    const auto length = GetSystemDirectoryW(system_directory.data(), system_directory.size());
    if (length == 0 || length >= system_directory.size()) {
      throw std::runtime_error("Unable to locate the Windows system directory");
    }
    return (std::filesystem::path(system_directory.data()) / "cmd.exe").string();
  }

  boost::process::v1::child exited_test_child() {
    boost::process::v1::child child {
      test_command_interpreter(),
      "/d",
      "/c",
      "exit /b 0",
      boost::process::v1::windows::create_no_window
    };
    // This build's Boost.Process wait() closes hProcess. Keep the terminated child's handle
    // valid, matching the production exit-observation path rather than an already-reaped child.
    if (WaitForSingleObject(child.native_handle(), 5000) != WAIT_OBJECT_0) {
      throw std::runtime_error("Test command did not exit before its deadline");
    }
    return child;
  }

  class IdleProcessLifecycleTest: public testing::Test {
  protected:
    void SetUp() override {
      stream::session::flush_platform_state();
      previous_process_ = std::move(proc::proc);
      proc::proc = proc::proc_t {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
      previous_driver_status_ = proc::vDisplayDriverStatus.exchange(VDISPLAY::DRIVER_STATUS::OK);
      previous_apps_path_ = config::stream.file_apps;
      previous_output_ = config::video.output_name;
      previous_grace_ = config::stream.session_resume_grace;
      config::stream.session_resume_grace = std::chrono::seconds {60};
      apps_path_ = std::filesystem::temp_directory_path() /
                   ("apollo_idle_process_test_" + std::to_string(GetCurrentProcessId()) + ".json");
      std::ofstream apps_file(apps_path_);
      ASSERT_TRUE(apps_file.is_open());
      apps_file << R"({"version":2,"env":{},"apps":[]})";
      config::stream.file_apps = apps_path_.string();

      original_ = std::make_shared<rtsp_stream::launch_session_t>();
      original_->id = 11;
      original_->width = 1920;
      original_->height = 1080;
      original_->fps = 60000;
      proc::process_test_access::retain(proc::proc, original_);
    }

    void TearDown() override {
      stream::session::flush_platform_state();
      proc::process_test_access::stop_child(proc::proc);
      proc::proc.terminate(false, false);
      proc::proc = std::move(previous_process_);
      proc::vDisplayDriverStatus.store(previous_driver_status_);
      config::stream.file_apps = previous_apps_path_;
      config::video.output_name = previous_output_;
      config::stream.session_resume_grace = previous_grace_;
      std::error_code error;
      std::filesystem::remove(apps_path_, error);
    }

    proc::proc_t previous_process_;
    VDISPLAY::DRIVER_STATUS previous_driver_status_;
    std::string previous_apps_path_;
    std::string previous_output_;
    std::chrono::milliseconds previous_grace_;
    std::filesystem::path apps_path_;
    std::shared_ptr<rtsp_stream::launch_session_t> original_;
  };

  class RetainedDisplayPauseTest: public testing::Test {
  protected:
    using operation_e = proc::display_topology_test_operation_e;

    void SetUp() override {
      saved_output_ = config::video.output_name;
      process_.initial_display = saved_output_;
      original_ = make_launch(11);
      original_->sbs_mode = 1;
      original_->display_guid.Data1 = 0x12345678;
      proc::process_test_access::retain(process_, original_);
      proc::process_test_access::mark_virtual(process_, true, true);
      proc::process_test_access::set_display_topology_hook(process_, [&](operation_e operation, bool) {
        operations_.push_back(operation);
        return failed_operation_ != operation;
      });
    }

    void TearDown() override {
      proc::process_test_access::clear(process_);
      config::video.output_name = saved_output_;
    }

    static std::shared_ptr<rtsp_stream::launch_session_t> make_launch(std::uint32_t id) {
      auto launch = std::make_shared<rtsp_stream::launch_session_t>();
      launch->id = id;
      launch->width = 1920;
      launch->height = 1080;
      launch->fps = 60000;
      launch->scale_factor = 100;
      launch->virtual_display = true;
      launch->virtual_display_only = true;
      return launch;
    }

    proc::proc_t process_ {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
    std::shared_ptr<rtsp_stream::launch_session_t> original_;
    std::vector<operation_e> operations_;
    std::optional<operation_e> failed_operation_;
    std::string saved_output_;
  };
}  // namespace

TEST_F(RetainedDisplayPauseTest, RestoresAndDetachesEveryVirtualPolicyBeforeReactivatingTheSameSession) {
  for (const bool exclusive : {false, true}) {
    SCOPED_TRACE(exclusive);
    original_->virtual_display_only = exclusive;
    proc::process_test_access::retain(process_, original_);
    operations_.clear();
    ASSERT_TRUE(process_.pause_display_for_resume());
    EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));
    EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
    EXPECT_EQ(process_.get_display_name(), "");
    EXPECT_EQ(process_.get_host_session_id(), 1234U);
    EXPECT_EQ(proc::process_test_access::virtual_device_path(process_), L"test-only-monitor-path");
    EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));

    operations_.clear();
    auto resumed = make_launch(22);
    resumed->virtual_display_only = exclusive;
    ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
    EXPECT_EQ(operations_, (std::vector<operation_e> {
                             operation_e::pause,
                             operation_e::reactivate,
                             operation_e::refresh_binding,
                             operation_e::request_hdr,
                             operation_e::promote,
                             operation_e::refresh_binding,
                           }));
    EXPECT_FALSE(proc::process_test_access::display_pause_pending(process_));
    EXPECT_EQ(process_.get_host_session_id(), 1234U);
    EXPECT_EQ(process_.get_status().app_id, 1);
    EXPECT_EQ(original_->id, 11U);
    EXPECT_EQ(resumed->display_guid.Data1, 0x12345678U);
    EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
    EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));
  }
}

TEST_F(RetainedDisplayPauseTest, FailedPhysicalRestorePreventsReactivationHdrAndTransportCommit) {
  failed_operation_ = operation_e::pause;
  ASSERT_FALSE(process_.pause_display_for_resume());
  operations_.clear();
  EXPECT_EQ(process_.reconfigure_retained_session(make_launch(22)), 503);
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::pause}));
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 11U);
  EXPECT_EQ(original_->sbs_mode, 1);
}

TEST_F(RetainedDisplayPauseTest, FailedReactivationLeavesExactSessionRetryable) {
  ASSERT_TRUE(process_.pause_display_for_resume());
  failed_operation_ = operation_e::reactivate;
  operations_.clear();
  auto resumed = make_launch(22);
  EXPECT_EQ(process_.reconfigure_retained_session(resumed), 503);
  EXPECT_EQ(operations_, (std::vector<operation_e> {
                           operation_e::pause,
                           operation_e::reactivate,
                           operation_e::pause,
                         }));
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 11U);
  EXPECT_EQ(original_->sbs_mode, 1);
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));

  failed_operation_.reset();
  operations_.clear();
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(operations_.front(), operation_e::pause);
  EXPECT_EQ(operations_[1], operation_e::reactivate);
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
  EXPECT_EQ(original_->sbs_mode, 0);
}

TEST_F(IdleProcessLifecycleTest, ExitedAppIsCleanedAfterMediaStopsBeforeWarmRetention) {
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  ASSERT_TRUE(proc::proc.stream_process_exited());
  ASSERT_EQ(proc::proc.get_status().app_id, 1);
  ASSERT_TRUE(stream::session::claim_active_slot_for_test());
  {
    auto release_active = util::fail_guard([]() {
      stream::session::release_active_slot_for_test();
    });
    auto guard = stream::session::guard_platform_launch();
    EXPECT_FALSE(guard.idle());
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  }
  stream::session::retain_or_stop_session_for_test(true);
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
}

TEST_F(IdleProcessLifecycleTest, PhysicalDesktopRestoresBeforeBoundedUndoCleanup) {
  const auto marker = std::filesystem::temp_directory_path() /
                      ("apollo_undo_order_" + std::to_string(GetCurrentProcessId()) + ".txt");
  std::filesystem::remove(marker);
  auto remove_marker = util::fail_guard([&]() {
    std::filesystem::remove(marker);
  });
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  proc::process_test_access::mark_virtual(proc::proc, true, true);
  original_->virtual_display = true;
  bool paused = false;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](auto operation, bool) {
    if (operation == proc::display_topology_test_operation_e::pause) {
      EXPECT_FALSE(std::filesystem::exists(marker)) << "Display recovery must precede undo commands";
      paused = true;
    }
    return true;
  });
  proc::process_test_access::add_undo(proc::proc, "\"" + test_command_interpreter() + "\" /d /c echo started>\"" + marker.string() + "\" & ping -n 30 127.0.0.1 >nul");
  const auto started = std::chrono::steady_clock::now();
  stream::session::retain_or_stop_session_for_test(true);
  EXPECT_TRUE(paused);
  EXPECT_TRUE(std::filesystem::exists(marker)) << "The test must actually execute the slow undo command";
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds {8});
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
}

TEST_F(IdleProcessLifecycleTest, IdleAdmissionReapsAnAppThatExitedDuringReconnectGrace) {
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  auto guard = stream::session::guard_platform_launch();
  EXPECT_TRUE(guard.idle());
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  EXPECT_EQ(proc::proc.reconfigure_retained_session(resumed), 409);
}

TEST_F(IdleProcessLifecycleTest, LiveCommandKeepsItsIdentityAndExistingGraceDeadline) {
  boost::process::v1::opstream child_input;
  boost::process::v1::child child {
    test_command_interpreter(),
    "/d",
    "/c",
    "set /p sunshine_test_wait=",
    boost::process::v1::std_in < child_input,
    boost::process::v1::windows::create_no_window
  };
  proc::process_test_access::set_child(proc::proc, std::move(child));
  stream::session::retain_or_stop_session_for_test(true);
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  ASSERT_TRUE(deadline);
  {
    auto guard = stream::session::guard_platform_launch();
    EXPECT_TRUE(guard.idle());
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  }
  EXPECT_EQ(stream::session::platform_lifecycle_generation_for_test(), generation);
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  proc::process_test_access::stop_child(proc::proc);
}

TEST_F(IdleProcessLifecycleTest, DesktopPlaceboSurvivesIdleAdmission) {
  proc::process_test_access::mark_desktop(proc::proc);
  auto guard = stream::session::guard_platform_launch();
  EXPECT_TRUE(guard.idle());
  EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::proc.get_status().app_id, 1);
}

TEST_F(IdleProcessLifecycleTest, GraceTaskReapsDisconnectedExitWithoutAnotherClientRequest) {
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  ASSERT_TRUE(stream::session::platform_stop_deadline_for_test());
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  stream::session::check_platform_stop_for_test();
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
  EXPECT_FALSE(stream::session::platform_stop_deadline_for_test());
}

TEST_F(IdleProcessLifecycleTest, GraceTaskPreservesDeadlineAndGenerationWhileAppLives) {
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  ASSERT_TRUE(deadline);
  for (int attempt = 0; attempt < 3; ++attempt) {
    stream::session::check_platform_stop_for_test();
    EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
    EXPECT_EQ(stream::session::platform_lifecycle_generation_for_test(), generation);
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  }
}

TEST_F(IdleProcessLifecycleTest, RepeatedDisconnectAndPauseFailureDoNotRenewGrace) {
  using operation_e = proc::display_topology_test_operation_e;
  proc::process_test_access::mark_desktop(proc::proc);
  proc::process_test_access::mark_virtual(proc::proc, true, true);
  bool pause_succeeds = false;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](operation_e operation, bool) {
    return operation != operation_e::pause || pause_succeeds;
  });
  stream::session::retain_or_stop_session_for_test(true);
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  ASSERT_TRUE(deadline);
  for (int attempt = 0; attempt < 3; ++attempt) {
    pause_succeeds = attempt == 2;
    stream::session::retain_or_stop_session_for_test(true);
    EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
    EXPECT_EQ(stream::session::platform_lifecycle_generation_for_test(), generation);
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
    EXPECT_TRUE(proc::process_test_access::has_virtual_identity(proc::proc));
  }
}

TEST_F(IdleProcessLifecycleTest, AuthorizedFreshLaunchReplacesRetainedAppBeforeGraceExpires) {
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  ASSERT_TRUE(stream::session::platform_stop_deadline_for_test());
  {
    auto fresh = stream::session::guard_platform_launch();
    ASSERT_TRUE(fresh.idle());
    ASSERT_TRUE(fresh.prepare_new_session());
    EXPECT_EQ(proc::proc.get_status().app_id, 0);
    EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
  }
  EXPECT_FALSE(stream::session::platform_stop_deadline_for_test());
}

TEST_F(IdleProcessLifecycleTest, OnlyFreshLocalConnectionMayReplaceRemoteGrace) {
  using admission_e = stream::session::local_session_admission_e;
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  EXPECT_EQ(stream::session::prepare_local_ar_session(false), admission_e::remote_busy);
  EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(stream::session::prepare_local_ar_session(true), admission_e::ready);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
  EXPECT_FALSE(stream::session::platform_stop_deadline_for_test());
}

TEST_F(IdleProcessLifecycleTest, FreshLocalConnectionReapsAnExitedIdleApp) {
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  EXPECT_EQ(stream::session::prepare_local_ar_session(true), stream::session::local_session_admission_e::ready);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
}

TEST_F(IdleProcessLifecycleTest, AcceptedHandshakeCannotBeReplacedByAnotherFreshConnection) {
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  {
    auto accepted = stream::session::guard_platform_launch();
    accepted.commit();
  }
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  {
    auto competing = stream::session::guard_platform_launch();
    EXPECT_FALSE(competing.prepare_new_session());
  }
  EXPECT_EQ(stream::session::prepare_local_ar_session(true), stream::session::local_session_admission_e::remote_busy);
  EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  EXPECT_EQ(stream::session::platform_lifecycle_generation_for_test(), generation);
}

TEST_F(IdleProcessLifecycleTest, PendingAndClaimedRtspReservationsRejectFreshReplacement) {
  proc::process_test_access::mark_desktop(proc::proc);
  stream::session::retain_or_stop_session_for_test(true);
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  for (const bool claimed : {false, true}) {
    SCOPED_TRACE(claimed);
    auto pending = std::make_shared<rtsp_stream::launch_session_t>();
    pending->id = claimed ? 91002 : 91001;
    pending->unique_id = "fresh-replacement-reservation-test";
    pending->gcm_key.assign(16, 0);
    pending->rtsp_cipher.emplace(pending->gcm_key, false);
    pending->av_ping_payload = "0123456789abcdef";
    auto clear_reservation = util::fail_guard([&]() {
      if (claimed) {
        rtsp_stream::finish_launch_session_for_test(*pending, false);
      }
      rtsp_stream::launch_session_clear(pending->id);
    });
    ASSERT_TRUE(rtsp_stream::launch_session_raise(pending));
    if (claimed) {
      ASSERT_TRUE(rtsp_stream::claim_launch_session_for_test(*pending));
    }
    {
      auto competing = stream::session::guard_platform_launch();
      EXPECT_FALSE(competing.prepare_new_session());
    }
    EXPECT_EQ(stream::session::prepare_local_ar_session(true), stream::session::local_session_admission_e::remote_busy);
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
    EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
  }
}

TEST_F(IdleProcessLifecycleTest, AutoDetachedCommandSurvivesIdleAdmission) {
  proc::process_test_access::set_child(proc::proc, exited_test_child(), true);
  auto guard = stream::session::guard_platform_launch();
  EXPECT_TRUE(guard.idle());
  EXPECT_FALSE(proc::proc.stream_process_exited());
  EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
}

TEST_F(IdleProcessLifecycleTest, WaitAllKeepsSessionWhileAnotherGroupMemberRuns) {
  boost::process::v1::opstream child_input;
  boost::process::v1::group group;
  boost::process::v1::child child {
    test_command_interpreter(),
    "/d",
    "/c",
    "set /p sunshine_test_wait=",
    group,
    boost::process::v1::std_in < child_input,
    boost::process::v1::windows::create_no_window
  };
  proc::process_test_access::set_child(proc::proc, exited_test_child());
  proc::process_test_access::wait_for_group(proc::proc, std::move(group));
  {
    auto guard = stream::session::guard_platform_launch();
    EXPECT_TRUE(guard.idle());
    EXPECT_FALSE(proc::proc.stream_process_exited());
    EXPECT_EQ(proc::proc.get_host_session_id(), 1234U);
  }
  child.terminate();
  child.wait();
}

TEST_F(IdleProcessLifecycleTest, FailedPhysicalHdrRollbackClearsUnprovenSession) {
  using operation_e = proc::display_topology_test_operation_e;
  std::vector<bool> requests;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](operation_e operation, bool desired) {
    EXPECT_EQ(operation, operation_e::request_hdr);
    requests.push_back(desired);
    return false;
  });
  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  resumed->id = 22;
  resumed->width = 1920;
  resumed->height = 1080;
  resumed->fps = 60000;
  resumed->scale_factor = 100;
  resumed->enable_hdr = true;
  EXPECT_EQ(proc::proc.reconfigure_retained_session(resumed), 503);
  EXPECT_EQ(requests, (std::vector<bool> {true, false}));
  EXPECT_EQ(proc::proc.get_status().app_id, 0);
  EXPECT_EQ(proc::proc.get_host_session_id(), 0U);
}

TEST(ProcessTest, PhysicalResumeAppliesChangedHdrBeforePublishingItsContract) {
  using operation_e = proc::display_topology_test_operation_e;
  for (const bool previous_hdr : {false, true}) {
    proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
    auto original = std::make_shared<rtsp_stream::launch_session_t>();
    original->id = 11;
    original->width = 1920;
    original->height = 1080;
    original->fps = 60000;
    original->enable_hdr = previous_hdr;
    proc::process_test_access::retain(process, original);
    auto cleanup = util::fail_guard([&]() {
      proc::process_test_access::clear(process);
    });
    std::vector<bool> requests;
    proc::process_test_access::set_display_topology_hook(process, [&](operation_e operation, bool desired) {
      EXPECT_EQ(operation, operation_e::request_hdr);
      EXPECT_EQ(process.get_status().enable_hdr, previous_hdr);
      requests.push_back(desired);
      return true;
    });
    auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
    resumed->id = 22;
    resumed->width = 1920;
    resumed->height = 1080;
    resumed->fps = 60000;
    resumed->scale_factor = 100;
    resumed->enable_hdr = !previous_hdr;
    EXPECT_EQ(process.reconfigure_retained_session(resumed), 0);
    EXPECT_EQ(requests, (std::vector<bool> {!previous_hdr}));
    EXPECT_EQ(process.get_status().enable_hdr, !previous_hdr);
    EXPECT_EQ(process.get_host_session_id(), 1234U);
    EXPECT_EQ(original->id, 11U);
  }
}

TEST(ProcessTest, FailedPhysicalHdrResumeRestoresPreviousHdrAndTransportIdentity) {
  using operation_e = proc::display_topology_test_operation_e;
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  proc::process_test_access::retain(process, original);
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });
  std::vector<bool> requests;
  proc::process_test_access::set_display_topology_hook(process, [&](operation_e operation, bool desired) {
    EXPECT_EQ(operation, operation_e::request_hdr);
    EXPECT_FALSE(process.get_status().enable_hdr);
    requests.push_back(desired);
    return !desired;
  });
  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  resumed->id = 22;
  resumed->width = 1920;
  resumed->height = 1080;
  resumed->fps = 60000;
  resumed->scale_factor = 100;
  resumed->enable_hdr = true;
  EXPECT_EQ(process.reconfigure_retained_session(resumed), 503);
  EXPECT_EQ(requests, (std::vector<bool> {true, false}));
  EXPECT_FALSE(process.get_status().enable_hdr);
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  proc::process_test_access::mark_virtual(process, true);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 11), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 22), proc::live_video_mode_result_e::needs_reconnect);
}

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

TEST(ProcessTest, RestorePrimaryDisplayPreservesRetainedVirtualDisplayForResume) {
  using operation_e = proc::display_topology_test_operation_e;

  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  original->virtual_display = true;
  original->virtual_display_only = true;
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

  ASSERT_TRUE(process.restore_primary_display());
  ASSERT_EQ(operations.size(), 2U);
  EXPECT_EQ(operations[0], (std::pair {operation_e::restore, true}));
  EXPECT_EQ(operations[1], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  EXPECT_TRUE(process.get_status().virtual_display);
  EXPECT_TRUE(proc::process_test_access::virtual_display_only(process));

  operations.clear();
  auto resumed = std::make_shared<rtsp_stream::launch_session_t>();
  resumed->id = 22;
  resumed->width = 1920;
  resumed->height = 1080;
  resumed->fps = 60000;
  resumed->scale_factor = 100;
  resumed->virtual_display_only = true;
  ASSERT_EQ(process.reconfigure_retained_session(resumed), 0);

  ASSERT_EQ(operations.size(), 4U);
  EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[1], (std::pair {operation_e::request_hdr, false}));
  EXPECT_EQ(operations[2], (std::pair {operation_e::promote, true}));
  EXPECT_EQ(operations[3], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  EXPECT_TRUE(process.get_status().virtual_display);
}

TEST(ProcessTest, RetainedVirtualDisplayResumeRetriesPromotionWhileWindowsTopologySettles) {
  using operation_e = proc::display_topology_test_operation_e;

  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto original = std::make_shared<rtsp_stream::launch_session_t>();
  original->id = 11;
  original->width = 1920;
  original->height = 1080;
  original->fps = 60000;
  original->sbs_mode = 1;
  original->virtual_display = true;
  original->virtual_display_only = true;
  proc::process_test_access::retain(process, original);
  proc::process_test_access::mark_virtual(process, true);
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });

  std::vector<std::pair<operation_e, bool>> operations;
  int promotion_attempts = 0;
  proc::process_test_access::set_display_topology_hook(
    process,
    [&](operation_e operation, bool value) {
      operations.emplace_back(operation, value);
      if (operation == operation_e::promote) {
        return ++promotion_attempts > 1;
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
  resumed->sbs_mode = 0;
  resumed->virtual_display_only = true;
  ASSERT_EQ(process.reconfigure_retained_session(resumed), 0);

  EXPECT_EQ(operations, (std::vector<std::pair<operation_e, bool>> {
                          {operation_e::refresh_binding, true},
                          {operation_e::request_hdr, false},
                          {operation_e::promote, true},
                          {operation_e::promote, true},
                          {operation_e::refresh_binding, true},
                        }));
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  EXPECT_EQ(process.get_status().app_id, 1);
  EXPECT_TRUE(process.get_status().virtual_display);
  EXPECT_EQ(original->sbs_mode, resumed->sbs_mode);
  EXPECT_EQ(original->id, 11U);
}

TEST(ProcessTest, FailedPostPromotionRebindPausesAndPreservesExactSessionForRetry) {
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
  bool fail_rebind = true;
  proc::process_test_access::set_display_topology_hook(
    process,
    [&](operation_e operation, bool value) {
      operations.emplace_back(operation, value);
      if (operation == operation_e::refresh_binding) {
        // The retained binding resolves before promotion, but its required post-promotion
        // verification fails after Windows accepted the topology change.
        return !fail_rebind || ++refresh_count == 1;
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
  // Restore the previous policy and pause the unchanged source instead of destroying the app
  // because Windows has temporarily failed to publish its promoted GDI binding.
  EXPECT_EQ(operations[4], (std::pair {operation_e::pause, false}));
  EXPECT_FALSE(proc::process_test_access::virtual_display_only(process));
  EXPECT_FALSE(original->virtual_display_only);
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  const auto status = process.get_status();
  EXPECT_EQ(status.app_id, 1);
  EXPECT_TRUE(status.virtual_display);
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process));
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process));
  EXPECT_EQ(proc::process_test_access::active_transport(process), 11U);

  fail_rebind = false;
  operations.clear();
  ASSERT_EQ(process.reconfigure_retained_session(resumed), 0);
  ASSERT_GE(operations.size(), 2U);
  EXPECT_EQ(operations[0].first, operation_e::pause);
  EXPECT_EQ(operations[1].first, operation_e::reactivate);
  EXPECT_EQ(process.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process), 22U);
  EXPECT_TRUE(proc::process_test_access::virtual_display_only(process));
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
  auto cleanup = util::fail_guard([&]() {
    proc::process_test_access::clear(process);
  });

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

TEST(ProcessTest, LocalArVirtualDisplayOnlyIsEnabledByDefault) {
  EXPECT_TRUE(config::default_local_ar_virtual_display_only);
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
  EXPECT_TRUE(proc::explorerRepairAllowedForRetirementForTest(true, true, true));

  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(false, true, true)) << "Warm retirement must retain debt without restarting Explorer";
  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(true, false, true)) << "The opt-in config must remain suppressible";
  EXPECT_FALSE(proc::explorerRepairAllowedForRetirementForTest(true, true, false)) << "Cleanup of a path that was never active creates no repair debt";
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
