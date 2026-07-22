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
#include "src/process.h"

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
