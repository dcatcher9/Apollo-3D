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
  // same-mode apply needs only the injected display binding, exercising transport admission itself.
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
      process._display_session.commit_resume();
      process._display_session.set_exclusive(launch->virtual_display_only);
#endif
    }
#ifdef _WIN32
    static void mark_virtual(
      proc_t &process, bool enabled, bool with_identity = false,
      std::function<VDISPLAY::display_identity_query_t()> identity_query = {}
    ) {
      // These owners contain only injected effects; consuming a fixture never removes a monitor.
      auto discarded = std::move(process._display_session);
      VDISPLAY::session_io_t io;
      io.restore_cursor = [](const auto &) {};
      io.pause = [&process](std::wstring_view, auto &retained, std::wstring_view) {
        const auto storage = std::make_shared<int>(0);
        retained = platf::primary_display::retained_display_ptr(
          storage,
          reinterpret_cast<platf::primary_display::retained_display_t *>(storage.get())
        );
        return !process._display_topology_test_hook || process._display_topology_test_hook(
                                                         display_topology_test_operation_e::pause,
                                                         process._virtual_display_only
                                                       );
      };
      io.reactivate = [&process](const auto &, bool exclusive) {
        return !process._display_topology_test_hook || process._display_topology_test_hook(
                                                         display_topology_test_operation_e::reactivate,
                                                         exclusive
                                                       );
      };
      io.query = [&process, identity_query](const auto &) {
        if (identity_query) {
          return identity_query();
        }
        const bool present = !process._display_topology_test_hook || process._display_topology_test_hook(
                                                                       display_topology_test_operation_e::refresh_binding,
                                                                       process._virtual_display_only
                                                                     );
        VDISPLAY::display_identity_query_t query;
        query.state = present ? VDISPLAY::display_identity_state_e::present : VDISPLAY::display_identity_state_e::indeterminate;
        if (present) {
          query.display_name = L"test-only-display";
          query.device_path = L"test-only-monitor-path";
        }
        return query;
      };
      process._display_session = VDISPLAY::session_t {std::move(io)};
      process._virtual_display = enabled;
      process.display_name.clear();
      process._display_mode_query_test_hook = {};
      process._display_mode_change_test_hook = {};
      process._display_mode_record_test_hook = {};
      process._display_baseline_test_hook = {};
      process._display_settings_unavailable_test_hook = {};
      process._saved_virtual_display_mode.reset();
      if (enabled) {
        if (!process._display_topology_test_hook) {
          process._display_topology_test_hook = [](auto, bool) { return true; };
        }
        // Fixtures never read or write Windows display settings; tests inspect saves explicitly.
        process._display_mode_record_test_hook = [](std::wstring_view, const auto &) {
          return DISP_CHANGE_SUCCESSFUL;
        };
        process._display_baseline_test_hook = [](std::wstring_view) {
          return true;
        };
        process._display_settings_unavailable_test_hook = [] {
          return false;
        };
        process._display_mode_query_test_hook = [&process](std::wstring_view) -> std::optional<DEVMODEW> {
          if (!process._launch_session) {
            return std::nullopt;
          }
          DEVMODEW mode {};
          mode.dmPelsWidth = process._launch_session->width;
          mode.dmPelsHeight = process._launch_session->height;
          mode.dmDisplayFrequency = (process._launch_session->fps + 500) / 1000;
          return mode;
        };
        VDISPLAY::creation_result_t binding;
        binding.display_name = L"test-only-display";
        binding.device_path = L"test-only-monitor-path";
        // Stable identity is required by the common owner even for same-mode test fixtures.
        binding.identity.emplace();
        VDISPLAY::display_spec_t spec;
        spec.exclusive = process._virtual_display_only;
        if (process._launch_session) {
          spec.guid = process._launch_session->display_guid;
        }
        process._display_session.adopt(std::move(spec), std::move(binding));
        // Like adopt_virtual_display(), a bound display publishes its capture target.
        process.display_name = "test-only-display";
      }
    }

    static bool virtual_display_only(const proc_t &process) {
      return process._virtual_display_only;
    }

    static bool display_pause_pending(const proc_t &process) {
      return process._display_session.pause_requested();
    }

    static bool has_retained_display(const proc_t &process) {
      return process._display_session.has_retained();
    }

    static std::wstring virtual_device_path(const proc_t &process) {
      return process._display_session.binding().device_path;
    }

    static bool has_virtual_identity(const proc_t &process) {
      return process._display_session.owns_display();
    }

    static std::uint32_t active_transport(const proc_t &process) {
      return process._active_launch_session_id;
    }

    static void observe_display_mode(proc_t &process, std::optional<DEVMODEW> mode) {
      process._display_mode_query_test_hook = [mode](std::wstring_view) { return mode; };
    }

    static void observe_display_mode(
      proc_t &process, std::function<std::optional<DEVMODEW>(std::wstring_view)> query
    ) {
      process._display_mode_query_test_hook = std::move(query);
    }

    static void configure_display_mode(
      proc_t &process, std::function<LONG(std::wstring_view, int, int, int, bool)> configure
    ) {
      process._display_mode_change_test_hook = std::move(configure);
    }

    static void record_session_mode(
      proc_t &process,
      std::function<LONG(std::wstring_view, const VDISPLAY::session_mode_record_t &)> record
    ) {
      process._display_mode_record_test_hook = std::move(record);
    }

    static void observe_baseline_topology(proc_t &process, std::function<bool(std::wstring_view)> query) {
      process._display_baseline_test_hook = std::move(query);
    }

    static void observe_display_settings_unavailable(proc_t &process, std::function<bool()> query) {
      process._display_settings_unavailable_test_hook = std::move(query);
    }

    static bool save_session_mode(proc_t &process, int width, int height, int fps_millihz) {
      return process.save_virtual_display_session_mode(width, height, fps_millihz);
    }

    static std::optional<std::array<int, 3>> saved_session_mode(const proc_t &process) {
      if (!process._saved_virtual_display_mode) {
        return std::nullopt;
      }
      const auto &mode = *process._saved_virtual_display_mode;
      return std::array<int, 3> {mode.width, mode.height, mode.fps_millihz};
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

    static void set_host_session_id(proc_t &process, std::uint64_t id) {
      process._host_session_id = id;
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
      process._display_session.commit_resume();
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
TEST(ProcessTest, RepeatedLiveRequestLeavesAnApplicationChosenDisplayModeAlone) {
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto launch = std::make_shared<rtsp_stream::launch_session_t>();
  launch->id = 41;
  launch->width = 3840;
  launch->height = 2160;
  launch->fps = 90000;
  launch->scale_factor = 100;
  proc::process_test_access::retain(process, launch);
  auto cleanup = util::fail_guard([&]() { proc::process_test_access::clear(process); });
  proc::process_test_access::mark_virtual(process, true);
  proc::process_test_access::set_display_topology_hook(process, [](auto, bool) { return true; });

  // An exclusive-fullscreen game moved the session display from 90 Hz to 72 Hz, and the client
  // reconfirms its unchanged 90 Hz request after the host rebuilt its encoder.
  int queries = 0;
  DEVMODEW observed {};
  observed.dmPelsWidth = 3840;
  observed.dmPelsHeight = 2160;
  observed.dmDisplayFrequency = 72;
  proc::process_test_access::observe_display_mode(process, [&](std::wstring_view) {
    ++queries;
    return std::optional {observed};
  });
  std::vector<std::pair<bool, int>> mode_requests;
  proc::process_test_access::configure_display_mode(process, [&](std::wstring_view, int, int, int fps, bool probe) {
    mode_requests.emplace_back(probe, fps);
    return DISP_CHANGE_SUCCESSFUL;
  });
  EXPECT_FALSE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 90000, 41), proc::live_video_mode_result_e::unchanged);
  // Neither the hint nor the apply queries, probes, or applies a display mode.
  EXPECT_EQ(queries, 0);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(launch->fps, 90000);  // The game never changed the accepted client request.

  // The request alone decides: another drifted geometry is left alone too.
  observed.dmPelsWidth = 1920;
  EXPECT_FALSE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 90000, 41), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 90000, 40), proc::live_video_mode_result_e::needs_reconnect);
  EXPECT_EQ(queries, 0);
  EXPECT_TRUE(mode_requests.empty());

  // Refresh rates are compared at the live wire's 0.01 Hz: any wire rate that differs is a real
  // change, while a launch rate finer than the wire is repeated by its nearest wire rate.
  launch->fps = 59940;
  EXPECT_FALSE(process.live_video_mode_needs_display_change(3840, 2160, 59940));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 59940, 41), proc::live_video_mode_result_e::unchanged);
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 60000));
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_TRUE(process.live_video_mode_needs_display_change(1920, 2160, 59940));
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 1080, 59940));
  for (const auto &[launch_fps, wire_fps] : {std::pair {59997, 60000}, std::pair {23976, 23980}, std::pair {119880, 119880}}) {
    SCOPED_TRACE(launch_fps);
    launch->fps = launch_fps;
    EXPECT_FALSE(process.live_video_mode_needs_display_change(3840, 2160, wire_fps));
    EXPECT_EQ(process.apply_live_video_mode(3840, 2160, wire_fps, 41), proc::live_video_mode_result_e::unchanged);
    EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, wire_fps + 10));
    EXPECT_EQ(launch->fps, launch_fps);
  }
  EXPECT_EQ(queries, 0);
  EXPECT_TRUE(mode_requests.empty());
}

TEST(ProcessTest, LiveModeResolvesOwnedIdentityBeforeUsingRenamedOrReusedDisplayNames) {
  const auto saved_output = config::video.output_name;
  auto restore_config = util::fail_guard([&]() {
    config::video.output_name = saved_output;
  });
  proc::proc_t process {boost::this_process::environment(), std::vector<proc::ctx_t> {}};
  auto launch = std::make_shared<rtsp_stream::launch_session_t>();
  launch->id = 41;
  launch->width = 3840;
  launch->height = 2160;
  launch->fps = 90000;
  launch->scale_factor = 100;
  proc::process_test_access::retain(process, launch);
  auto cleanup = util::fail_guard([&]() { proc::process_test_access::clear(process); });
  bool owned_present = true;
  std::wstring owned_name = L"renamed-owned-display";
  proc::process_test_access::mark_virtual(process, true, true, [&]() {
    VDISPLAY::display_identity_query_t identity;
    identity.state = owned_present ? VDISPLAY::display_identity_state_e::present : VDISPLAY::display_identity_state_e::indeterminate;
    identity.display_name = owned_name;
    identity.device_path = L"test-only-monitor-path";
    return identity;
  });
  proc::process_test_access::set_display_topology_hook(process, [](auto, bool) { return true; });
  unsigned observed_queries = 0;
  DWORD owned_hz = 72;
  proc::process_test_access::observe_display_mode(process, [&](std::wstring_view name) -> std::optional<DEVMODEW> {
    ++observed_queries;
    EXPECT_EQ(name, owned_name);
    DEVMODEW mode {};
    mode.dmPelsWidth = 3840;
    mode.dmPelsHeight = 2160;
    // The old GDI name has been reused by a different monitor already at 90 Hz.
    mode.dmDisplayFrequency = name == owned_name ? owned_hz : 90;
    return mode;
  });
  std::vector<std::wstring> configured_names;
  proc::process_test_access::configure_display_mode(process, [&](std::wstring_view name, int, int, int fps, bool probe) {
    configured_names.emplace_back(name);
    if (!probe && name == owned_name) {
      owned_hz = static_cast<DWORD>(fps / 1000);
    }
    return DISP_CHANGE_SUCCESSFUL;
  });

  // Windows renumbered the owned display. The hint resolves the new name, but a stale transport's
  // request is refused before publishing it; the published capture target stays stale.
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 90000, 40), proc::live_video_mode_result_e::needs_reconnect);
  EXPECT_EQ(process.get_display_name(), "test-only-display");
  // The next repeated request still differs from the published target, so it publishes the
  // resolved binding without reading or setting a mode through either name.
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 90000, 41), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.get_display_name(), "renamed-owned-display");
  EXPECT_FALSE(process.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(observed_queries, 0u);
  EXPECT_TRUE(configured_names.empty());
  EXPECT_EQ(owned_hz, 72u);

  // A changed request probes, applies, and verifies only through the resolved owned name.
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 60000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 60000, 41), proc::live_video_mode_result_e::applied);
  EXPECT_EQ(configured_names, (std::vector<std::wstring> {owned_name, owned_name}));
  EXPECT_GT(observed_queries, 0u);
  EXPECT_EQ(owned_hz, 60u);
  EXPECT_EQ(launch->fps, 60000);

  // Neither a repeated nor a changed request is acknowledged or probed through the stale GDI name
  // of a vanished/unresolved owned identity.
  owned_present = false;
  const auto prior_queries = observed_queries;
  configured_names.clear();
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 60000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 60000, 41), proc::live_video_mode_result_e::failed);
  EXPECT_EQ(process.apply_live_video_mode(1920, 1080, 60000, 41), proc::live_video_mode_result_e::failed);
  EXPECT_EQ(observed_queries, prior_queries);
  EXPECT_TRUE(configured_names.empty());

  // A later resolution of the same exact owner may recover under another name.
  owned_present = true;
  owned_name = L"renamed-again-owned-display";
  EXPECT_TRUE(process.live_video_mode_needs_display_change(3840, 2160, 60000));
  EXPECT_EQ(process.apply_live_video_mode(3840, 2160, 60000, 41), proc::live_video_mode_result_e::unchanged);
  EXPECT_EQ(process.get_display_name(), "renamed-again-owned-display");
  EXPECT_EQ(observed_queries, prior_queries);
  EXPECT_TRUE(configured_names.empty());
  EXPECT_EQ(launch->fps, 60000);
}

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
    EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::wake}));
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
                             operation_e::reactivate,
                             operation_e::refresh_binding,
                             operation_e::refresh_binding,
                             operation_e::request_hdr,
                             operation_e::promote,
                             operation_e::refresh_binding,
                           }));
    EXPECT_FALSE(proc::process_test_access::display_pause_pending(process_));
    EXPECT_FALSE(proc::process_test_access::has_retained_display(process_));
    EXPECT_EQ(process_.get_host_session_id(), 1234U);
    EXPECT_EQ(process_.get_status().app_id, 1);
    EXPECT_EQ(original_->id, 11U);
    EXPECT_EQ(resumed->display_guid.Data1, 0x12345678U);
    EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
    EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));
  }
}

TEST_F(RetainedDisplayPauseTest, ResumeRepairsObservedFullscreenModeWithoutChangingRequestedContract) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  for (const auto observed : {std::array<DWORD, 3> {4800, 2700, 90}, std::array<DWORD, 3> {3840, 2160, 72}}) {
    SCOPED_TRACE(observed[0]);
    SCOPED_TRACE(observed[2]);
    DEVMODEW current {};
    current.dmPelsWidth = observed[0];
    current.dmPelsHeight = observed[1];
    current.dmDisplayFrequency = observed[2];
    proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view name) {
      EXPECT_EQ(name, L"test-only-display");
      return std::optional {current};
    });
    std::vector<bool> mode_requests;
    proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view name, int width, int height, int fps, bool probe) {
      EXPECT_EQ(name, L"test-only-display");
      EXPECT_EQ(width, 3840);
      EXPECT_EQ(height, 2160);
      EXPECT_EQ(fps, 90000);
      mode_requests.push_back(probe);
      if (!probe) {
        current.dmPelsWidth = width;
        current.dmPelsHeight = height;
        current.dmDisplayFrequency = fps / 1000;
      }
      return DISP_CHANGE_SUCCESSFUL;
    });
    ASSERT_TRUE(process_.pause_display_for_resume());
    auto resumed = make_launch(22);
    resumed->width = original_->width;
    resumed->height = original_->height;
    resumed->fps = original_->fps;
    ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
    EXPECT_EQ(mode_requests, (std::vector<bool> {true, false}));
    EXPECT_EQ(current.dmPelsWidth, 3840U);
    EXPECT_EQ(current.dmPelsHeight, 2160U);
    EXPECT_EQ(current.dmDisplayFrequency, 90U);
    EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
    EXPECT_EQ(process_.get_host_session_id(), 1234U);
    EXPECT_FALSE(proc::process_test_access::display_pause_pending(process_));
  }
}

TEST_F(RetainedDisplayPauseTest, UnchangedResumeChecksActualModeWithoutResettingIt) {
  // The same integer-Hz tolerance as live changes accepts fractional requested refresh rates.
  original_->fps = 59940;
  int queries = 0;
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    ++queries;
    DEVMODEW current {};
    current.dmPelsWidth = 1920;
    current.dmPelsHeight = 1080;
    current.dmDisplayFrequency = 60;
    return std::optional {current};
  });
  int mode_requests = 0;
  proc::process_test_access::configure_display_mode(process_, [&](auto, auto, auto, auto, auto) {
    ++mode_requests;
    return DISP_CHANGE_SUCCESSFUL;
  });
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto resumed = make_launch(22);
  resumed->fps = original_->fps;
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(queries, 1);
  EXPECT_EQ(mode_requests, 0);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
}

TEST_F(RetainedDisplayPauseTest, UnobservableUnchangedModeKeepsResumeRetryableWithoutReplacingTheDisplay) {
  proc::process_test_access::observe_display_mode(process_, std::nullopt);
  int mode_requests = 0;
  proc::process_test_access::configure_display_mode(process_, [&](auto, auto, auto, auto, auto) {
    ++mode_requests;
    return DISP_CHANGE_SUCCESSFUL;
  });
  ASSERT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(process_.reconfigure_retained_session(make_launch(22)), 503);
  EXPECT_EQ(mode_requests, 0);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 11U);
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::retire), 0);
}

namespace {
  struct saved_session_mode_t {
    std::wstring display_name;
    VDISPLAY::session_mode_record_t record;
  };

  void expect_virtual_display_only_record(const saved_session_mode_t &saved, DWORD width, DWORD height, DWORD refresh_hz) {
    // One device, resolution and refresh only: no position/primary, no other display, no apply.
    EXPECT_EQ(saved.display_name, L"test-only-display");
    EXPECT_EQ(saved.record.width, width);
    EXPECT_EQ(saved.record.height, height);
    EXPECT_EQ(saved.record.refresh_hz, refresh_hz);
    EXPECT_EQ(saved.record.fields, static_cast<DWORD>(DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY));
    EXPECT_EQ(saved.record.flags, static_cast<DWORD>(CDS_UPDATEREGISTRY | CDS_GLOBAL | CDS_NORESET));
  }
}  // namespace

TEST_F(RetainedDisplayPauseTest, ResumeSavesTheSessionModeBeforePromotionForTheVirtualDisplayOnly) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 72;  // A fullscreen game left another rate in place.
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    return std::optional {current};
  });
  std::vector<bool> mode_requests;
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view, int width, int height, int fps, bool probe) {
    mode_requests.push_back(probe);
    if (!probe) {
      current.dmPelsWidth = width;
      current.dmPelsHeight = height;
      current.dmDisplayFrequency = fps / 1000;
    }
    return DISP_CHANGE_SUCCESSFUL;
  });
  std::vector<std::wstring> baseline_checks;
  proc::process_test_access::observe_baseline_topology(process_, [&](std::wstring_view device_path) {
    baseline_checks.emplace_back(device_path);
    return true;
  });
  std::vector<saved_session_mode_t> saves;
  std::vector<std::ptrdiff_t> promotions_before_save;
  proc::process_test_access::record_session_mode(process_, [&](std::wstring_view name, const VDISPLAY::session_mode_record_t &record) {
    saves.push_back({std::wstring {name}, record});
    promotions_before_save.push_back(std::ranges::count(operations_, operation_e::promote));
    return DISP_CHANGE_SUCCESSFUL;
  });

  ASSERT_TRUE(process_.pause_display_for_resume());
  operations_.clear();
  auto resumed = make_launch(22);
  resumed->width = 3840;
  resumed->height = 2160;
  resumed->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(mode_requests, (std::vector<bool> {true, false}));
  ASSERT_EQ(saves.size(), 1u);
  expect_virtual_display_only_record(saves.front(), 3840, 2160, 90);
  // Saved while the reactivated display still extended the user's own desktop.
  EXPECT_EQ(promotions_before_save, (std::vector<std::ptrdiff_t> {0}));
  EXPECT_EQ(baseline_checks, (std::vector<std::wstring> {L"test-only-monitor-path"}));
  EXPECT_EQ(proc::process_test_access::saved_session_mode(process_), (std::array<int, 3> {3840, 2160, 90000}));
  // Saving is not a topology operation: resume asked the topology owner only for its own steps.
  EXPECT_EQ(std::ranges::count(operations_, operation_e::promote), 1);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::restore), 0);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::retire), 0);

  // A later drift is repaired on resume without rewriting the already saved default.
  saves.clear();
  baseline_checks.clear();
  mode_requests.clear();
  current.dmDisplayFrequency = 72;
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto drifted = make_launch(33);
  drifted->width = 3840;
  drifted->height = 2160;
  drifted->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(drifted), 0);
  EXPECT_EQ(mode_requests, (std::vector<bool> {true, false}));
  EXPECT_EQ(current.dmDisplayFrequency, 90u);
  EXPECT_TRUE(saves.empty());
  EXPECT_TRUE(baseline_checks.empty());

  // An unchanged resume verifies the mode but does not save.
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto unchanged = make_launch(44);
  unchanged->width = 3840;
  unchanged->height = 2160;
  unchanged->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(unchanged), 0);
  EXPECT_TRUE(saves.empty());
}

TEST_F(RetainedDisplayPauseTest, SessionTopologyOrRejectedSaveKeepsResumeTemporaryAndNonFatal) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 72;
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    return std::optional {current};
  });
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view, int width, int height, int fps, bool probe) {
    if (!probe) {
      current.dmPelsWidth = width;
      current.dmPelsHeight = height;
      current.dmDisplayFrequency = fps / 1000;
    }
    return DISP_CHANGE_SUCCESSFUL;
  });
  bool user_topology = false;
  proc::process_test_access::observe_baseline_topology(process_, [&](std::wstring_view) {
    return user_topology;
  });
  int writes = 0;
  LONG save_status = DISP_CHANGE_SUCCESSFUL;
  proc::process_test_access::record_session_mode(process_, [&](std::wstring_view, const auto &) {
    ++writes;
    return save_status;
  });

  // Windows restored a remembered session topology (virtual primary or physical outputs off):
  // nothing is written, and the resume still applies its temporary mode.
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto first = make_launch(22);
  first->width = 3840;
  first->height = 2160;
  first->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(first), 0);
  EXPECT_EQ(current.dmDisplayFrequency, 90u);
  EXPECT_EQ(writes, 0);
  EXPECT_FALSE(proc::process_test_access::saved_session_mode(process_));

  // Windows rejects the write: it is logged, and the resume still succeeds at the session mode.
  user_topology = true;
  save_status = DISP_CHANGE_NOTUPDATED;
  current.dmDisplayFrequency = 72;
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto second = make_launch(33);
  second->width = 3840;
  second->height = 2160;
  second->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(second), 0);
  EXPECT_EQ(current.dmDisplayFrequency, 90u);
  EXPECT_EQ(writes, 1);
  EXPECT_FALSE(proc::process_test_access::saved_session_mode(process_));
}

TEST_F(RetainedDisplayPauseTest, LiveModeChangesFollowRequestEdgesAndStayTemporary) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  proc::process_test_access::retain(process_, original_);
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 90;
  int queries = 0;
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view name) {
    EXPECT_EQ(name, L"test-only-display");
    ++queries;
    return std::optional {current};
  });
  std::vector<std::pair<bool, int>> mode_requests;
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view name, int width, int height, int fps, bool probe) {
    EXPECT_EQ(name, L"test-only-display");
    mode_requests.emplace_back(probe, fps);
    if (!probe) {
      current.dmPelsWidth = width;
      current.dmPelsHeight = height;
      current.dmDisplayFrequency = fps / 1000;
    }
    return DISP_CHANGE_SUCCESSFUL;
  });
  int saves = 0;
  proc::process_test_access::observe_baseline_topology(process_, [&](std::wstring_view) {
    ++saves;
    return true;
  });
  proc::process_test_access::record_session_mode(process_, [&](std::wstring_view, const auto &) {
    ++saves;
    return DISP_CHANGE_SUCCESSFUL;
  });
  using result_e = proc::live_video_mode_result_e;
  const std::vector<std::pair<bool, int>> applied_72 {{true, 72000}, {false, 72000}};
  const std::vector<std::pair<bool, int>> applied_90 {{true, 90000}, {false, 90000}};

  // The headset panel moved to 72 Hz: a verified temporary apply on the promoted display.
  EXPECT_TRUE(process_.live_video_mode_needs_display_change(3840, 2160, 72000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 72000, 11), result_e::applied);
  EXPECT_EQ(mode_requests, applied_72);
  EXPECT_EQ(original_->fps, 72000);

  // An exclusive-fullscreen application then moved the display to another rate. The client's
  // repeated request is not a mode change: it is acknowledged without querying, probing, or
  // applying a mode, and the application keeps the mode it chose.
  current.dmDisplayFrequency = 60;
  mode_requests.clear();
  queries = 0;
  operations_.clear();
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(3840, 2160, 72000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 72000, 11), result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(queries, 0);
  EXPECT_EQ(current.dmDisplayFrequency, 60u);
  EXPECT_EQ(original_->fps, 72000);
  // Only the exact binding is refreshed: no HDR request or topology promotion either.
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::refresh_binding, operation_e::refresh_binding}));

  // A real change applies as before, from whatever mode the application left in place.
  EXPECT_TRUE(process_.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 90000, 11), result_e::applied);
  EXPECT_EQ(mode_requests, applied_90);
  EXPECT_EQ(current.dmDisplayFrequency, 90u);
  EXPECT_EQ(original_->fps, 90000);
  // The promoted display's mode is never saved: not even the topology gate is consulted.
  EXPECT_EQ(saves, 0);
  EXPECT_FALSE(proc::process_test_access::saved_session_mode(process_));

  // A locked session keeps a real change retryable and leaves the display untouched.
  mode_requests.clear();
  current.dmDisplayFrequency = 60;
  proc::process_test_access::observe_display_settings_unavailable(process_, [] {
    return true;
  });
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 72000, 11), result_e::failed);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(current.dmDisplayFrequency, 60u);
  EXPECT_EQ(original_->fps, 90000);
  EXPECT_EQ(saves, 0);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::restore), 0);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::retire), 0);
}

TEST_F(RetainedDisplayPauseTest, ResumeRestoresTheRequestedModeThatRepeatedLiveRequestsLeaveAlone) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  proc::process_test_access::retain(process_, original_);
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 72;  // An exclusive-fullscreen game chose 72 Hz.
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    return std::optional {current};
  });
  std::vector<std::pair<bool, int>> mode_requests;
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view, int width, int height, int fps, bool probe) {
    mode_requests.emplace_back(probe, fps);
    if (!probe) {
      current.dmPelsWidth = width;
      current.dmPelsHeight = height;
      current.dmDisplayFrequency = fps / 1000;
    }
    return DISP_CHANGE_SUCCESSFUL;
  });

  // Live: the client's repeated 90 Hz request does not push the game out of its mode.
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 90000, 11), proc::live_video_mode_result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(current.dmDisplayFrequency, 72u);

  // A resume is a new start: the same requested mode is applied to the drifted display.
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto resumed = make_launch(22);
  resumed->width = 3840;
  resumed->height = 2160;
  resumed->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(mode_requests, (std::vector<std::pair<bool, int>> {{true, 90000}, {false, 90000}}));
  EXPECT_EQ(current.dmDisplayFrequency, 90u);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
}

TEST_F(RetainedDisplayPauseTest, UnprovenLiveChangeRequiresReconnectForEveryRequestUntilResume) {
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 90000;
  proc::process_test_access::retain(process_, original_);
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 90;
  int queries = 0;
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    ++queries;
    return std::optional {current};
  });
  std::vector<int> failing_sets;
  std::vector<std::pair<bool, int>> mode_requests;
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view, int width, int height, int fps, bool probe) {
    mode_requests.emplace_back(probe, fps);
    if (probe) {
      return DISP_CHANGE_SUCCESSFUL;
    }
    if (std::ranges::find(failing_sets, fps) != failing_sets.end()) {
      return DISP_CHANGE_FAILED;
    }
    current.dmPelsWidth = width;
    current.dmPelsHeight = height;
    current.dmDisplayFrequency = fps / 1000;
    return DISP_CHANGE_SUCCESSFUL;
  });
  using result_e = proc::live_video_mode_result_e;

  // A failed change with a proven rollback stays retryable, and the edge rule still applies.
  failing_sets = {72000};
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 72000, 11), result_e::failed);
  EXPECT_EQ(mode_requests, (std::vector<std::pair<bool, int>> {{true, 72000}, {false, 72000}, {false, 90000}}));
  mode_requests.clear();
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 90000, 11), result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());

  // Windows accepts the probe but neither the 72 Hz set nor the rollback to 90 Hz: the display is
  // unproven, the session keeps its requested mode, and the client is told to reconnect.
  failing_sets = {72000, 90000};
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 72000, 11), result_e::needs_reconnect);
  EXPECT_EQ(mode_requests, (std::vector<std::pair<bool, int>> {{true, 72000}, {false, 72000}, {false, 90000}}));
  EXPECT_EQ(original_->fps, 90000);

  // Neither a repeated request of the session's mode nor a changed one is acknowledged on that
  // display, and neither queries, probes, nor sets a mode.
  mode_requests.clear();
  queries = 0;
  EXPECT_TRUE(process_.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 90000, 11), result_e::needs_reconnect);
  EXPECT_TRUE(process_.live_video_mode_needs_display_change(3840, 2160, 60000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 60000, 11), result_e::needs_reconnect);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(queries, 0);

  // The client reconnects. The resume proves the display again, and the edge rule resumes.
  failing_sets.clear();
  ASSERT_TRUE(process_.pause_display_for_resume());
  auto resumed = make_launch(22);
  resumed->width = 3840;
  resumed->height = 2160;
  resumed->fps = 90000;
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  mode_requests.clear();
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(3840, 2160, 90000));
  EXPECT_EQ(process_.apply_live_video_mode(3840, 2160, 90000, 22), result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());
}

TEST_F(RetainedDisplayPauseTest, LiveRequestsUseTheSessionRenderScaleAndWirePrecision) {
  // The client streams 2560x1440 at a 150% render scale, and its launch rate is finer than the
  // live wire's 0.01 Hz.
  original_->width = 3840;
  original_->height = 2160;
  original_->fps = 59997;
  original_->scale_factor = 150;
  proc::process_test_access::retain(process_, original_);
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 60;
  proc::process_test_access::observe_display_mode(process_, [&](std::wstring_view) {
    return std::optional {current};
  });
  std::vector<std::array<int, 4>> mode_requests;
  proc::process_test_access::configure_display_mode(process_, [&](std::wstring_view, int width, int height, int fps, bool probe) {
    mode_requests.push_back({probe ? 1 : 0, width, height, fps});
    if (!probe) {
      current.dmPelsWidth = width;
      current.dmPelsHeight = height;
      current.dmDisplayFrequency = fps / 1000;
    }
    return DISP_CHANGE_SUCCESSFUL;
  });
  using result_e = proc::live_video_mode_result_e;

  // The client's reconfirm of its stream mode repeats the session's requested mode.
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(2560, 1440, 60000));
  EXPECT_EQ(process_.apply_live_video_mode(2560, 1440, 60000, 11), result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());
  EXPECT_EQ(original_->width, 3840);
  EXPECT_EQ(original_->fps, 59997);

  // A real change renders at the same scale: a 1920x1080 stream gets a 2880x1620 display.
  EXPECT_TRUE(process_.live_video_mode_needs_display_change(1920, 1080, 60000));
  EXPECT_EQ(process_.apply_live_video_mode(1920, 1080, 60000, 11), result_e::applied);
  EXPECT_EQ(mode_requests, (std::vector<std::array<int, 4>> {{1, 2880, 1620, 60000}, {0, 2880, 1620, 60000}}));
  EXPECT_EQ(original_->width, 2880);
  EXPECT_EQ(original_->height, 1620);
  EXPECT_EQ(original_->fps, 60000);

  // Its own reconfirm is again a repeat.
  mode_requests.clear();
  EXPECT_FALSE(process_.live_video_mode_needs_display_change(1920, 1080, 60000));
  EXPECT_EQ(process_.apply_live_video_mode(1920, 1080, 60000, 11), result_e::unchanged);
  EXPECT_TRUE(mode_requests.empty());
}

TEST_F(RetainedDisplayPauseTest, UnverifiedSessionModeIsNeverSaved) {
  int saves = 0;
  proc::process_test_access::record_session_mode(process_, [&](std::wstring_view, const auto &) {
    ++saves;
    return DISP_CHANGE_SUCCESSFUL;
  });
  DEVMODEW current {};
  current.dmPelsWidth = 3840;
  current.dmPelsHeight = 2160;
  current.dmDisplayFrequency = 60;
  proc::process_test_access::observe_display_mode(process_, std::optional {current});
  EXPECT_FALSE(proc::process_test_access::save_session_mode(process_, 3840, 2160, 90000));
  proc::process_test_access::observe_display_mode(process_, std::nullopt);
  EXPECT_FALSE(proc::process_test_access::save_session_mode(process_, 3840, 2160, 90000));
  // DEVMODE frequency 1 means "hardware default", which is not a session rate.
  current.dmDisplayFrequency = 1;
  proc::process_test_access::observe_display_mode(process_, std::optional {current});
  EXPECT_FALSE(proc::process_test_access::save_session_mode(process_, 3840, 2160, 1000));
  EXPECT_EQ(saves, 0);
  EXPECT_FALSE(proc::process_test_access::saved_session_mode(process_));

  current.dmDisplayFrequency = 90;
  proc::process_test_access::observe_display_mode(process_, std::optional {current});
  EXPECT_TRUE(proc::process_test_access::save_session_mode(process_, 3840, 2160, 90000));
  EXPECT_EQ(saves, 1);
  // The same verified mode is already the saved default.
  EXPECT_TRUE(proc::process_test_access::save_session_mode(process_, 3840, 2160, 90000));
  EXPECT_EQ(saves, 1);
}

TEST(ProcessDisplayModeContract, SessionModeIsSavedOnlyBeforePromotionAndModeSetsStayTemporary) {
  std::ifstream input(SUNSHINE_SOURCE_DIR "/src/process.cpp", std::ios::binary);
  ASSERT_TRUE(input.is_open());
  const std::string source {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  constexpr std::string_view save_call = "save_virtual_display_session_mode(";

  // Creation saves the verified session mode right after applying it, before topology promotion.
  const auto created = source.find("Virtual Display created at");
  ASSERT_NE(created, std::string::npos);
  const auto applied = source.find("VDISPLAY::changeDisplaySettings(created_display.display_name.c_str()", created);
  const auto saved = source.find("save_virtual_display_session_mode(render_width, render_height, target_fps);", created);
  const auto promoted = source.find("promote_virtual_display(launch_session->enable_hdr)", created);
  ASSERT_NE(applied, std::string::npos);
  ASSERT_NE(saved, std::string::npos);
  ASSERT_NE(promoted, std::string::npos);
  EXPECT_LT(applied, saved);
  EXPECT_LT(saved, promoted);

  // Resume saves a changed mode before its own promotion.
  const auto resume = source.find("int proc_t::reconfigure_retained_session(");
  ASSERT_NE(resume, std::string::npos);
  const auto resume_save = source.find(save_call, resume);
  const auto resume_promote = source.find("promote_virtual_display(launch_session->enable_hdr, 1500ms)", resume);
  ASSERT_NE(resume_save, std::string::npos);
  ASSERT_NE(resume_promote, std::string::npos);
  EXPECT_LT(resume_save, resume_promote);

  // Live changes run on the promoted display and never save.
  const auto live = source.find("live_video_mode_result_e proc_t::apply_live_video_mode(");
  const auto live_end = source.find("bool proc_t::activate_remote_virtual_display_lease(", live);
  ASSERT_NE(live, std::string::npos);
  ASSERT_NE(live_end, std::string::npos);
  EXPECT_EQ(source.substr(live, live_end - live).find(save_call), std::string::npos);

  // Live requests are edge-triggered: neither the hint nor the apply restores a drifted mode. Only
  // a resume, which is a new start, reports restoring the requested mode after drift.
  const auto hint = source.find("bool proc_t::live_video_mode_needs_display_change(");
  ASSERT_NE(hint, std::string::npos);
  ASSERT_LT(hint, live);
  EXPECT_EQ(source.substr(hint, live_end - hint).find("Restoring the requested"), std::string::npos);
  EXPECT_NE(source.substr(resume, hint - resume).find("Restoring the requested virtual-display mode on resume"), std::string::npos);

  // Exactly those two call sites plus the definition exist.
  std::size_t save_mentions = 0;
  for (auto at = source.find(save_call); at != std::string::npos; at = source.find(save_call, at + 1)) {
    ++save_mentions;
  }
  EXPECT_EQ(save_mentions, 3u);

  // Every host mode set is temporary: SDC_SAVE_TO_DATABASE would save the whole topology,
  // including physical displays disabled by a virtual-display-only session.
  std::size_t mode_sets = 0;
  for (auto call = source.find("VDISPLAY::changeDisplaySettings("); call != std::string::npos; call = source.find("VDISPLAY::changeDisplaySettings(", call + 1)) {
    ++mode_sets;
    const auto end = source.find(')', source.find("fps", call));
    ASSERT_NE(end, std::string::npos);
    EXPECT_EQ(source.substr(end - 7, 8), ", false)") << source.substr(call, end - call + 1);
  }
  EXPECT_EQ(mode_sets, 2u);
  EXPECT_EQ(source.find("SDC_SAVE_TO_DATABASE"), std::string::npos);
  EXPECT_EQ(source.find("CDS_UPDATEREGISTRY"), std::string::npos);
}

TEST_F(RetainedDisplayPauseTest, FailedPhysicalRestorePreventsReactivationHdrAndTransportCommit) {
  failed_operation_ = operation_e::pause;
  ASSERT_FALSE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));
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
                           operation_e::reactivate,
                           operation_e::pause,
                           operation_e::wake,
                         }));
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 11U);
  EXPECT_EQ(original_->sbs_mode, 1);
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));

  failed_operation_.reset();
  operations_.clear();
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_EQ(operations_.front(), operation_e::reactivate);
  EXPECT_EQ(operations_[1], operation_e::refresh_binding);
  EXPECT_EQ(process_.get_host_session_id(), 1234U);
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
  EXPECT_EQ(original_->sbs_mode, 0);
}

TEST_F(RetainedDisplayPauseTest, FailedHdrAfterReactivationKeepsRecoveryUntilSuccessfulRetry) {
  ASSERT_TRUE(process_.pause_display_for_resume());
  bool fail_hdr = true;
  operations_.clear();
  proc::process_test_access::set_display_topology_hook(process_, [&](operation_e operation, bool) {
    operations_.push_back(operation);
    if (operation == operation_e::request_hdr || operation == operation_e::promote) {
      EXPECT_TRUE(proc::process_test_access::has_retained_display(process_));
    }
    if (operation == operation_e::request_hdr && fail_hdr) {
      fail_hdr = false;  // The requested HDR fails; restoring the prior HDR still succeeds.
      return false;
    }
    return true;
  });
  auto resumed = make_launch(22);
  EXPECT_EQ(process_.reconfigure_retained_session(resumed), 503);
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_retained_display(process_));
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 11U);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::reactivate), 1);
  EXPECT_EQ(std::ranges::count(operations_, operation_e::wake), 1);
  operations_.clear();
  EXPECT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  EXPECT_FALSE(proc::process_test_access::has_retained_display(process_));
  EXPECT_EQ(proc::process_test_access::active_transport(process_), 22U);
}

TEST_F(RetainedDisplayPauseTest, RequestsWakeOnceAfterSuccessfulRecoveryIncludingRetry) {
  failed_operation_ = operation_e::pause;
  EXPECT_FALSE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));

  failed_operation_.reset();
  operations_.clear();
  ASSERT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::wake}));

  operations_.clear();
  EXPECT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));
  operations_.clear();
  process_.terminate(false, false);
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::retire}));
  EXPECT_EQ(process_.get_host_session_id(), 0U);
}

TEST_F(RetainedDisplayPauseTest, RecheckFailureDoesNotRepeatWakeButCompletedResumeStartsANewPauseEpisode) {
  ASSERT_TRUE(process_.pause_display_for_resume());
  operations_.clear();
  failed_operation_ = operation_e::pause;
  EXPECT_FALSE(process_.pause_display_for_resume());
  failed_operation_.reset();
  ASSERT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::pause}));

  auto resumed = make_launch(22);
  ASSERT_EQ(process_.reconfigure_retained_session(resumed), 0);
  operations_.clear();
  ASSERT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::wake}));
}

TEST_F(RetainedDisplayPauseTest, RejectedWakeDoesNotInvalidateRecoveredTopologyOrRepeatOnCleanup) {
  failed_operation_ = operation_e::wake;
  ASSERT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::wake}));
  EXPECT_TRUE(proc::process_test_access::display_pause_pending(process_));
  EXPECT_TRUE(proc::process_test_access::has_retained_display(process_));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(process_));
  EXPECT_EQ(process_.get_host_session_id(), 1234U);

  operations_.clear();
  EXPECT_TRUE(process_.pause_display_for_resume());
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause}));
  operations_.clear();
  process_.terminate(false, false);
  EXPECT_EQ(operations_, (std::vector<operation_e> {operation_e::pause, operation_e::retire}));
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

TEST_F(IdleProcessLifecycleTest, FailedRetirementPreservesOwnerAcrossRefreshAndNewLaunch) {
  using operation_e = proc::display_topology_test_operation_e;
  original_->virtual_display = true;
  proc::process_test_access::mark_virtual(proc::proc, true, true);
  bool allow_retirement = false;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](operation_e operation, bool) {
    return operation != operation_e::retire || allow_retirement;
  });
  auto cleanup = util::fail_guard([&]() {
    // The injected binding must never escape this fixture, including on an assertion failure.
    proc::process_test_access::clear(proc::proc);
  });
  const auto device_path = proc::process_test_access::virtual_device_path(proc::proc);

  proc::proc.terminate(false, false);
  ASSERT_TRUE(proc::process_test_access::has_virtual_identity(proc::proc));
  EXPECT_EQ(proc::proc.get_status().app_id, 0);

  EXPECT_NO_THROW(proc::refresh(apps_path_.string(), false));
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(proc::proc));
  EXPECT_EQ(proc::process_test_access::virtual_device_path(proc::proc), device_path);

  proc::ctx_t next_app;
  auto next_launch = std::make_shared<rtsp_stream::launch_session_t>();
  // Cleanup admission precedes even input validation. Invalid geometry also keeps this negative
  // test away from platform acquisition if the guard regresses.
  next_launch->width = next_launch->height = 1;
  next_launch->scale_factor = 20;
  EXPECT_EQ(proc::proc.execute(next_app, next_launch, false), 503);
  EXPECT_TRUE(proc::process_test_access::has_virtual_identity(proc::proc));
  EXPECT_EQ(proc::process_test_access::virtual_device_path(proc::proc), device_path);

  allow_retirement = true;
  proc::proc.terminate(false, false);
  EXPECT_FALSE(proc::process_test_access::has_virtual_identity(proc::proc));
  EXPECT_NO_THROW(proc::refresh(apps_path_.string(), false));
  EXPECT_FALSE(proc::process_test_access::has_virtual_identity(proc::proc));
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

TEST_F(IdleProcessLifecycleTest, GraceTaskRechecksSuccessfulPauseWithoutRenewingDeadlineOrWake) {
  using operation_e = proc::display_topology_test_operation_e;
  proc::process_test_access::mark_desktop(proc::proc);
  proc::process_test_access::mark_virtual(proc::proc, true, true);
  unsigned pauses = 0;
  unsigned wakes = 0;
  bool target_active = true;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](operation_e operation, bool) {
    if (operation == operation_e::pause) {
      ++pauses;
      target_active = false;
    } else if (operation == operation_e::wake) {
      ++wakes;
    }
    return true;
  });
  stream::session::retain_or_stop_session_for_test(true);
  ASSERT_EQ(pauses, 1u);
  ASSERT_EQ(wakes, 1u);
  const auto deadline = stream::session::platform_stop_deadline_for_test();
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  ASSERT_TRUE(deadline);
  for (int attempt = 0; attempt < 3; ++attempt) {
    target_active = true;  // Simulates an owned retained device reappearing after success.
    stream::session::check_platform_stop_for_test();
    EXPECT_FALSE(target_active);
    EXPECT_EQ(stream::session::platform_stop_deadline_for_test(), deadline);
    EXPECT_EQ(stream::session::platform_lifecycle_generation_for_test(), generation);
  }
  EXPECT_EQ(pauses, 4u);
  EXPECT_EQ(wakes, 1u);
  EXPECT_EQ(proc::proc.get_host_session_id(), 1234u);
}

TEST_F(IdleProcessLifecycleTest, GracePauseCheckCannotTouchAnActiveOrReplacementSession) {
  using operation_e = proc::display_topology_test_operation_e;
  proc::process_test_access::mark_desktop(proc::proc);
  proc::process_test_access::mark_virtual(proc::proc, true, true);
  unsigned pauses = 0;
  proc::process_test_access::set_display_topology_hook(proc::proc, [&](operation_e operation, bool) {
    pauses += operation == operation_e::pause;
    return true;
  });
  stream::session::retain_or_stop_session_for_test(true);
  ASSERT_EQ(pauses, 1u);
  const auto generation = stream::session::platform_lifecycle_generation_for_test();
  ASSERT_TRUE(stream::session::claim_active_slot_for_test());
  stream::session::check_platform_stop_for_test();
  stream::session::release_active_slot_for_test();
  EXPECT_EQ(pauses, 1u);

  proc::process_test_access::set_host_session_id(proc::proc, 9999);
  stream::session::check_platform_stop_for_test();
  EXPECT_EQ(pauses, 1u);
  proc::process_test_access::set_host_session_id(proc::proc, 1234);
  {
    auto guard = stream::session::guard_platform_launch();
    guard.commit();  // An accepted launch changes generation before RTSP starts.
  }
  ASSERT_NE(stream::session::platform_lifecycle_generation_for_test(), generation);
  stream::session::check_platform_stop_for_test(generation);
  stream::session::check_platform_stop_for_test();  // Connecting is not retained/paused.
  EXPECT_EQ(pauses, 1u);
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
  original->scale_factor = 100;
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
  // The following virtual-display transport check also resolves its owned
  // identity; the physical HDR-only observation above is complete.
  proc::process_test_access::set_display_topology_hook(process, [](auto, bool) { return true; });
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

    ASSERT_EQ(operations.size(), 5U);
    EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, !previous_policy}));
    EXPECT_EQ(operations[1], (std::pair {operation_e::refresh_binding, !previous_policy}));
    EXPECT_EQ(operations[2], (std::pair {operation_e::request_hdr, false}));
    EXPECT_EQ(operations[3], (std::pair {operation_e::promote, !previous_policy}));
    EXPECT_EQ(operations[4], (std::pair {operation_e::refresh_binding, !previous_policy}));
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

  ASSERT_EQ(operations.size(), 5U);
  EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[1], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[2], (std::pair {operation_e::request_hdr, false}));
  EXPECT_EQ(operations[3], (std::pair {operation_e::promote, true}));
  EXPECT_EQ(operations[4], (std::pair {operation_e::refresh_binding, true}));
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
        // The retained binding and actual-mode query resolve before promotion, but its required
        // post-promotion verification fails after Windows accepted the topology change.
        return !fail_rebind || ++refresh_count <= 2;
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

  ASSERT_EQ(operations.size(), 7U);
  EXPECT_EQ(operations[0], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[1], (std::pair {operation_e::refresh_binding, true}));
  EXPECT_EQ(operations[2], (std::pair {operation_e::request_hdr, false}));
  EXPECT_EQ(operations[3], (std::pair {operation_e::promote, true}));
  EXPECT_EQ(operations[4], (std::pair {operation_e::refresh_binding, true}));
  // Restore the previous policy and pause the unchanged source instead of destroying the app
  // because Windows has temporarily failed to publish its promoted GDI binding.
  EXPECT_EQ(operations[5], (std::pair {operation_e::pause, false}));
  EXPECT_EQ(operations[6], (std::pair {operation_e::wake, false}));
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
  EXPECT_EQ(operations[0].first, operation_e::reactivate);
  EXPECT_EQ(operations[1].first, operation_e::refresh_binding);
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

  // The live-mode worker's hint must say "no display work" so a bitrate-only change is
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
